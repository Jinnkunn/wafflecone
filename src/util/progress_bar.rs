
use indicatif;
use crate::util::constant;

pub struct ProgressBar {
    bar: Option<indicatif::ProgressBar>,
    if_show: bool,
}

impl ProgressBar {
    pub fn new(total: u64, title: &str, if_show: bool) -> ProgressBar {
        match if_show {
            true => {
                let bar = indicatif::ProgressBar::new(total);
                bar.set_style(
                    indicatif::ProgressStyle::with_template(constant::TEMPLATE)
                        .unwrap()
                        .progress_chars(constant::PROGRESS_CHARS),
                );
                bar.set_prefix(title.to_string());
                ProgressBar {
                    bar: Some(bar),
                    if_show,
                }
            }
            false => {
                ProgressBar {
                    bar: None,
                    if_show,
                }
            }
        }
    }

    pub fn inc(&mut self, n: u64) {
        if self.if_show {
            self.bar.as_ref().unwrap().inc(n);
        }
    }

    pub fn finish(&mut self) {
        if self.if_show {
            self.bar.as_ref().unwrap().finish();
        }
    }
}