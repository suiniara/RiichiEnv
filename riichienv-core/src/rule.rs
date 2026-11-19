#[cfg(feature = "python")]
use pyo3::{pyclass, pymethods};
use serde::{Deserialize, Serialize};

fn default_true() -> bool {
    true
}

#[cfg_attr(
    feature = "python",
    pyclass(module = "riichienv._riichienv", get_all, set_all, from_py_object)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GameRule {
    pub allows_ron_on_ankan_for_kokushi_musou: bool,
    pub is_kokushi_musou_13machi_double: bool,
    pub is_suuankou_tanki_double: bool,
    pub is_junsei_chuurenpoutou_double: bool,
    pub is_daisuushii_double: bool,
    pub yakuman_pao_is_liability_only: bool,
    pub sanchaho_is_draw: bool,

    pub kuikae_forbidden: bool,

    /// Treat the dealer's uninterrupted initial 14 tiles as hand tiles when
    /// recording a discard. This does not change the first-draw win context.
    #[serde(default)]
    pub dealer_first_discard_is_tedashi: bool,

    /// Tenhou gives dealer continuation priority in extension rounds;
    /// Mahjong Soul keeps only its explicit dealer multi-ron exception.
    #[serde(default = "default_true")]
    pub extension_dealer_repeat_priority: bool,

    /// Tenhou admits formal waits when four owned copies include a meld;
    /// Mahjong Soul excludes a wait already occupying four owned copies.
    #[serde(default = "default_true")]
    pub allow_fifth_copy_tenpai: bool,
}

impl Default for GameRule {
    fn default() -> Self {
        Self::default_tenhou()
    }
}

impl GameRule {
    pub fn default_tenhou() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: false,
            is_kokushi_musou_13machi_double: false,
            is_suuankou_tanki_double: false,
            is_junsei_chuurenpoutou_double: false,
            is_daisuushii_double: false,
            yakuman_pao_is_liability_only: false,

            sanchaho_is_draw: true,

            kuikae_forbidden: true,
            dealer_first_discard_is_tedashi: false,
            extension_dealer_repeat_priority: true,
            allow_fifth_copy_tenpai: true,
        }
    }

    pub fn default_mjsoul() -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou: true,
            is_kokushi_musou_13machi_double: true,
            is_suuankou_tanki_double: true,
            is_junsei_chuurenpoutou_double: true,
            is_daisuushii_double: true,
            yakuman_pao_is_liability_only: true,

            sanchaho_is_draw: false,

            kuikae_forbidden: true,
            dealer_first_discard_is_tedashi: true,
            extension_dealer_repeat_priority: false,
            allow_fifth_copy_tenpai: false,
        }
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl GameRule {
    #[new]
    #[pyo3(signature = (allows_ron_on_ankan_for_kokushi_musou=false, is_kokushi_musou_13machi_double=false, is_suuankou_tanki_double=false, is_junsei_chuurenpoutou_double=false, is_daisuushii_double=false, yakuman_pao_is_liability_only=false, sanchaho_is_draw=false, kuikae_forbidden=true, dealer_first_discard_is_tedashi=false, extension_dealer_repeat_priority=true, allow_fifth_copy_tenpai=true))]
    #[allow(clippy::too_many_arguments)]
    pub fn py_new(
        allows_ron_on_ankan_for_kokushi_musou: bool,
        is_kokushi_musou_13machi_double: bool,
        is_suuankou_tanki_double: bool,
        is_junsei_chuurenpoutou_double: bool,
        is_daisuushii_double: bool,
        yakuman_pao_is_liability_only: bool,
        sanchaho_is_draw: bool,
        kuikae_forbidden: bool,
        dealer_first_discard_is_tedashi: bool,
        extension_dealer_repeat_priority: bool,
        allow_fifth_copy_tenpai: bool,
    ) -> Self {
        Self {
            allows_ron_on_ankan_for_kokushi_musou,
            is_kokushi_musou_13machi_double,
            is_suuankou_tanki_double,
            is_junsei_chuurenpoutou_double,
            is_daisuushii_double,
            yakuman_pao_is_liability_only,
            sanchaho_is_draw,
            kuikae_forbidden,
            dealer_first_discard_is_tedashi,
            extension_dealer_repeat_priority,
            allow_fifth_copy_tenpai,
        }
    }

    #[staticmethod]
    #[pyo3(name = "default_tenhou")]
    pub fn py_default_tenhou() -> Self {
        Self::default_tenhou()
    }

    #[staticmethod]
    #[pyo3(name = "default_mjsoul")]
    pub fn py_default_mjsoul() -> Self {
        Self::default_mjsoul()
    }

    fn __repr__(&self) -> String {
        format!(
            "GameRule(allows_ron_on_ankan_for_kokushi_musou={}, is_kokushi_musou_13machi_double={}, is_suuankou_tanki_double={}, is_junsei_chuurenpoutou_double={}, is_daisuushii_double={}, yakuman_pao_is_liability_only={}, sanchaho_is_draw={}, kuikae_forbidden={}, dealer_first_discard_is_tedashi={}, extension_dealer_repeat_priority={}, allow_fifth_copy_tenpai={})",
            self.allows_ron_on_ankan_for_kokushi_musou,
            self.is_kokushi_musou_13machi_double,
            self.is_suuankou_tanki_double,
            self.is_junsei_chuurenpoutou_double,
            self.is_daisuushii_double,
            self.yakuman_pao_is_liability_only,
            self.sanchaho_is_draw,
            self.kuikae_forbidden,
            self.dealer_first_discard_is_tedashi,
            self.extension_dealer_repeat_priority,
            self.allow_fifth_copy_tenpai
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_defaults_and_round_trip_keep_the_rule_differences() {
        let default = GameRule::default();
        assert!(default.extension_dealer_repeat_priority);
        assert!(default.allow_fifth_copy_tenpai);
        let mjsoul = GameRule::default_mjsoul();
        assert!(!mjsoul.extension_dealer_repeat_priority);
        assert!(!mjsoul.allow_fifth_copy_tenpai);
        let restored: GameRule =
            serde_json::from_value(serde_json::to_value(mjsoul).unwrap()).unwrap();
        assert_eq!(restored, mjsoul);
    }

    #[test]
    fn legacy_serialization_keeps_the_old_engine_semantics() {
        let mut legacy = serde_json::to_value(GameRule::default_tenhou()).unwrap();
        let fields = legacy.as_object_mut().unwrap();
        fields.remove("extension_dealer_repeat_priority");
        fields.remove("allow_fifth_copy_tenpai");
        let restored: GameRule = serde_json::from_value(legacy).unwrap();
        assert!(restored.extension_dealer_repeat_priority);
        assert!(restored.allow_fifth_copy_tenpai);
    }
}
