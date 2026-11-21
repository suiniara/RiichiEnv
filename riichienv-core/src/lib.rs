pub mod agari;
pub mod errors;
pub mod hand_evaluator;
pub mod hand_evaluator_3p;
pub mod score;
mod tests;
pub mod types;
pub mod v2_features;
pub mod yaku;
mod yaku_3p;

pub mod action;
pub mod game_variant;
pub mod observation;
pub mod observation_3p;
pub mod observer;
pub mod parser;
pub mod public_rules;
pub mod replay;
pub mod replay_context;
pub mod rule;
pub mod rules;
pub mod shanten;
pub mod state;
pub mod state_3p;
#[cfg(feature = "python")]
mod yaku_checker;

pub use hand_evaluator::check_riichi_candidates;

/// Shared production rule/observer contract, exposed to offline data artifacts.
pub const NATIVE_RULES_VERSION: &str = "riichienv_native_rules_v4_nagashi_mask";
