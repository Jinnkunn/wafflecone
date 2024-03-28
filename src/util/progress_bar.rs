use crate::util::constant;
use indicatif;

pub struct ProgressBar {
    bar: Option<indicatif::ProgressBar>,
}

impl ProgressBar {
    pub fn new(total: u64, title: &str, if_show: bool) -> ProgressBar {
        let bar = if if_show {
            let bar = indicatif::ProgressBar::new(total);
            bar.set_style(
                indicatif::ProgressStyle::with_template(constant::TEMPLATE)
                    .unwrap()
                    .progress_chars(constant::PROGRESS_CHARS),
            );
            bar.set_prefix(title.to_string());
            Some(bar)
        } else {
            None
        };

        ProgressBar { bar }
    }

    pub fn inc(&mut self, n: u64) {
        if let Some(bar) = &self.bar {
            bar.inc(n);
        }
    }

    pub fn finish(&mut self) {
        if let Some(bar) = &self.bar {
            bar.finish();
        }
    }
}
