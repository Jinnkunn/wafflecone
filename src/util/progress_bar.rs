
use indicatif;

pub struct ProgressBar {
    bar: Option<indicatif::ProgressBar>,
    if_show: bool,
}

impl ProgressBar {
    pub fn new(total: u64, if_show: bool) -> ProgressBar {
        if if_show {
            ProgressBar {
                bar: Some(indicatif::ProgressBar::new(total)),
                if_show,
            }
        } else {
            ProgressBar {
                bar: None,
                if_show,
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