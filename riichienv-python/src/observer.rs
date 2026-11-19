use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use riichienv_core::observer::{ObserverState, pending_kan_commits};

#[pyclass(module = "riichienv._riichienv")]
pub struct NativeObserverState {
    state: ObserverState,
}

#[pymethods]
impl NativeObserverState {
    #[new]
    fn new(observer: usize) -> PyResult<Self> {
        Ok(Self {
            state: ObserverState::new(observer).map_err(PyValueError::new_err)?,
        })
    }

    fn apply_event_json(&mut self, event: &str) -> PyResult<String> {
        let event =
            serde_json::from_str(event).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let public = self
            .state
            .apply_event(&event)
            .map_err(PyValueError::new_err)?;
        serde_json::to_string(&public).map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn state_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.state).map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyfunction]
fn native_pending_kan_commits(
    event: &str,
    actor: usize,
    committed_kans: i64,
    indicators: i64,
) -> PyResult<bool> {
    let event = serde_json::from_str(event).map_err(|e| PyValueError::new_err(e.to_string()))?;
    if actor >= 4 {
        return Err(PyValueError::new_err(
            "Kan actor must be a four-player seat",
        ));
    }
    pending_kan_commits(&event, actor, committed_kans, indicators).map_err(PyValueError::new_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeObserverState>()?;
    m.add_function(wrap_pyfunction!(native_pending_kan_commits, m)?)?;
    Ok(())
}
