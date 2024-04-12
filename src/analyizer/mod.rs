use crate::space::space_generator::Space;

pub mod calculator;

pub trait SpaceCalculator {
    fn new(model_name: String, bias_free_token_space: Space, bias_group_spaces: Vec<Space>)
        -> Self;
}
