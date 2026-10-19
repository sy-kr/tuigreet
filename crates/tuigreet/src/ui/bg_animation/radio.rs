//! Radio waves: concentric rings expanding from the screen centre.
//!
//! A port of the "ASCII Radio Waves" animation from asciiart.eu. Each cell's
//! brightness is a Gaussian falloff from the nearest ring crest, dimmed with
//! distance from the centre, with a little per-cell noise on top. Brightness
//! selects both the glyph (sparse to dense) and the colour (dark to bright);
//! cells below a visibility threshold are left untouched so the regular
//! background shows through.
//!
//! Rings move in wall-clock time, so the frame rate affects smoothness but
//! not how fast the waves travel.

use std::time::Instant;

use tui::{
  buffer::Buffer,
  layout::{Position, Rect},
  style::Color,
};

use super::Animation;

/// Cells dimmer than this are not drawn at all.
const VISIBLE: f32 = 0.05;

/// Gamma applied to brightness before glyph / colour lookup; lifts mid-tones.
const GAMMA: f32 = 0.75;

/// Crest travel speed, as a fraction of the wavelength per second.
const CREST_SPEED: f32 = 0.65;

/// How many times per second the noise pattern re-rolls.
const NOISE_RATE: f64 = 12.0;

/// Longest frame interval honoured when advancing time, so a stalled
/// terminal resumes smoothly instead of jumping.
const MAX_DT: f32 = 0.05;

/// Configurable parameters for the radio waves effect.
#[derive(Debug, Clone)]
pub struct Options {
  /// Animation speed multiplier.
  pub speed:       f32,
  /// Distance between ring crests, in cells.
  pub wavelength:  f32,
  /// Width of each ring, as a fraction of the wavelength.
  pub thickness:   f32,
  /// How strongly rings fade with distance from the centre. 0 disables it.
  pub decay:       f32,
  /// Per-cell brightness jitter, 0..=1.
  pub noise:       f32,
  /// Height of a terminal cell relative to its width; keeps rings round.
  pub cell_aspect: f32,
  /// Glyphs from dimmest to brightest. The last one is only reached when a
  /// cell saturates, which near the centre happens on random frames thanks
  /// to noise, so keep it something that can flicker without jarring.
  pub glyphs:      Vec<char>,
  /// Colours from dimmest to brightest.
  pub palette:     Vec<Color>,
}

impl Default for Options {
  fn default() -> Self {
    Self {
      speed:       1.15,
      wavelength:  13.0,
      thickness:   0.27,
      decay:       1.0,
      noise:       0.1,
      cell_aspect: 2.0,
      glyphs:      vec![' ', '.', ',', ':', ';', 'i', 'l', 'I'],
      palette:     vec![
        Color::Rgb(0x00, 0x1A, 0x4D),
        Color::Rgb(0x00, 0x3D, 0x99),
        Color::Rgb(0x00, 0x66, 0xFF),
        Color::Rgb(0x66, 0xB2, 0xFF),
        Color::Rgb(0xE6, 0xF3, 0xFF),
      ],
    }
  }
}

pub struct Radio {
  width:  u16,
  height: u16,
  /// Distance of every cell from the centre, in cell widths, row-major.
  dist:   Vec<f32>,
  /// Per-cell fade rate: `decay` spread over the half-diagonal.
  fade:   f32,
  opts:   Options,
  /// Animation time in seconds, already scaled by `speed`.
  t:      f64,
  last:   Option<Instant>,
}

/// Per-frame values shared by every cell.
struct Frame {
  /// How far the crests have travelled, wrapped to one wavelength.
  crest:       f32,
  /// Which roll of the noise pattern is showing.
  noise_frame: u32,
}

impl Radio {
  pub fn new(mut opts: Options) -> Self {
    opts.wavelength = opts.wavelength.max(1.0);
    opts.thickness = opts.thickness.max(1e-3);
    opts.noise = opts.noise.clamp(0.0, 1.0);
    opts.cell_aspect = opts.cell_aspect.clamp(0.5, 4.0);
    if opts.glyphs.is_empty() {
      opts.glyphs = Options::default().glyphs;
    }
    if opts.palette.is_empty() {
      opts.palette = Options::default().palette;
    }

    Self {
      width: 0,
      height: 0,
      dist: Vec::new(),
      fade: 0.0,
      opts,
      t: 0.0,
      last: None,
    }
  }

  /// Advance the animation by `dt` wall-clock seconds.
  pub fn advance(&mut self, dt: f32) {
    self.t += f64::from(dt * self.opts.speed);
  }

  #[expect(clippy::cast_possible_truncation, reason = "value is wrapped")]
  #[expect(clippy::cast_sign_loss, reason = "time never goes negative")]
  fn frame(&self) -> Frame {
    let wavelength = f64::from(self.opts.wavelength);
    let travelled = self.t * f64::from(CREST_SPEED) * wavelength;
    Frame {
      crest:       (travelled % wavelength) as f32,
      noise_frame: (self.t * NOISE_RATE) as u32,
    }
  }

  /// Raw brightness of one cell, before the visibility cut and gamma.
  fn cell(&self, x: u16, y: u16, frame: &Frame) -> f32 {
    let idx = usize::from(y) * usize::from(self.width) + usize::from(x);
    let dist = self.dist[idx];
    let phase = (dist - frame.crest) / self.opts.wavelength;
    let phase = phase - phase.floor();
    let to_crest = phase.min(1.0 - phase);
    let thickness = self.opts.thickness;
    let ring = (-to_crest * to_crest / (thickness * thickness)).exp();
    let mut bright = ring * (-dist * self.fade).exp();
    if self.opts.noise > 0.0 {
      let jitter = 2.0f32.mul_add(hash(x, y, frame.noise_frame), -1.0);
      bright *= jitter.mul_add(self.opts.noise, 1.0);
    }
    bright
  }

  /// Raw brightness of one cell at the current time.
  #[cfg(test)]
  fn brightness(&self, x: u16, y: u16) -> f32 {
    self.cell(x, y, &self.frame())
  }
}

/// Pick the entry of `items` that `v` (0..=1) maps to.
#[expect(clippy::cast_possible_truncation, reason = "index is clamped")]
#[expect(clippy::cast_precision_loss, reason = "tiny slice lengths")]
#[expect(clippy::cast_sign_loss, reason = "v is clamped to 0..=1")]
fn pick<T>(items: &[T], v: f32) -> &T {
  let last = items.len() - 1;
  let idx = (v.clamp(0.0, 1.0) * last as f32) as usize;
  &items[idx.min(last)]
}

/// Cheap per-cell hash in 0..=1, re-rolled whenever `frame` changes.
#[expect(clippy::cast_precision_loss, reason = "only need ~24 bits")]
fn hash(x: u16, y: u16, frame: u32) -> f32 {
  let mut s = u32::from(x).wrapping_mul(374_761_393)
    ^ u32::from(y).wrapping_mul(668_265_263)
    ^ frame.wrapping_mul(2_147_483_647);
  s = (s ^ (s >> 13)).wrapping_mul(1_274_126_177);
  s ^= s >> 16;
  s as f32 / u32::MAX as f32
}

impl Animation for Radio {
  fn resize(&mut self, area: Rect) {
    if area.width == self.width
      && area.height == self.height
      && !self.dist.is_empty()
    {
      return;
    }
    self.width = area.width;
    self.height = area.height;

    let (w, h) = (f32::from(area.width), f32::from(area.height));
    let (cx, cy) = (0.5 * (w - 1.0), 0.5 * (h - 1.0));
    self.fade = self.opts.decay / (0.5 * w).hypot(0.5 * h).max(1.0);

    self.dist.clear();
    for y in 0..area.height {
      let dy = (f32::from(y) - cy) * self.opts.cell_aspect;
      for x in 0..area.width {
        self.dist.push((f32::from(x) - cx).hypot(dy));
      }
    }
  }

  fn step(&mut self) {
    let now = Instant::now();
    if let Some(last) = self.last {
      self.advance(now.duration_since(last).as_secs_f32().min(MAX_DT));
    }
    self.last = Some(now);
  }

  fn render(&self, area: Rect, buf: &mut Buffer) {
    if self.width == 0 || self.height == 0 {
      return;
    }
    let frame = self.frame();

    for ly in 0..self.height {
      for lx in 0..self.width {
        let v = self.cell(lx, ly, &frame);
        if v < VISIBLE {
          continue;
        }
        let v = v.min(1.0).powf(GAMMA);
        let pos = Position {
          x: area.x + lx,
          y: area.y + ly,
        };
        if let Some(cell) = buf.cell_mut(pos) {
          cell.set_char(*pick(&self.opts.glyphs, v));
          cell.set_fg(*pick(&self.opts.palette, v));
        }
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use tui::{
    buffer::Buffer,
    layout::{Position, Rect},
  };

  use super::*;

  /// 41x21 puts the centre exactly on cell (20, 10).
  fn sized(opts: Options) -> Radio {
    let mut r = Radio::new(opts);
    r.resize(Rect::new(0, 0, 41, 21));
    r
  }

  /// Brightness of each cell on the centre row, from the centre rightwards.
  fn centre_row(r: &Radio) -> Vec<f32> {
    (20..41).map(|x| r.brightness(x, 10)).collect()
  }

  #[test]
  fn centre_is_brightest_at_start() {
    // Noise off: it may legitimately lift a neighbour above the centre.
    let r = sized(Options {
      noise: 0.0,
      ..Options::default()
    });
    let centre = r.brightness(20, 10);
    for y in 0..21 {
      for x in 0..41 {
        assert!(
          r.brightness(x, y) <= centre,
          "cell ({x},{y}) outshines centre"
        );
      }
    }
  }

  #[test]
  fn rings_travel_outward_with_time() {
    let opts = Options {
      noise: 0.0,
      ..Options::default()
    };
    let mut r = sized(opts.clone());
    // Crests move at 0.65 * wavelength cells per second of animation time.
    // Advance far enough for the first crest to sit 5 cells out.
    let crest_speed = 0.65 * opts.wavelength;
    r.advance(5.0 / crest_speed / opts.speed);

    let row = centre_row(&r);
    let peak = (0..row.len())
      .max_by(|&a, &b| row[a].total_cmp(&row[b]))
      .unwrap();
    assert!((4..=6).contains(&peak), "crest at offset {peak}, expected ~5");
  }

  #[test]
  fn resize_rebuilds_distance_cache() {
    let mut r = Radio::new(Options::default());
    r.resize(Rect::new(0, 0, 8, 4));
    assert_eq!(r.dist.len(), 32);
    r.resize(Rect::new(0, 0, 16, 8));
    assert_eq!(r.dist.len(), 128);
  }

  #[test]
  fn render_paints_crests_and_leaves_troughs_alone() {
    let opts = Options {
      noise: 0.0,
      ..Options::default()
    };
    let r = sized(opts.clone());
    let area = Rect::new(0, 0, 41, 21);
    let mut buf = Buffer::empty(area);

    r.render(area, &mut buf);

    // Centre sits on a crest: brightest glyph in the brightest colour.
    let centre = &buf[Position { x: 20, y: 10 }];
    assert_eq!(centre.symbol(), opts.glyphs.last().unwrap().to_string());
    assert_eq!(centre.fg, *opts.palette.last().unwrap());

    // Halfway between crests (wavelength / 2 ≈ 6.5 cells) is below the
    // visibility threshold, so the cell must be untouched.
    let trough = &buf[Position { x: 26, y: 10 }];
    assert_eq!(trough.symbol(), " ");
    assert_eq!(trough.fg, tui::style::Color::Reset);
  }
}
