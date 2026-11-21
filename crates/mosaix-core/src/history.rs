//! Snapshot-based undo history. Snapshots are whole documents, which are
//! cheap because tile maps are shared (`Arc`) between steps.

pub struct History<S> {
    undo: Vec<(S, String)>,
    redo: Vec<(S, String)>,
    limit: usize,
}

impl<S> History<S> {
    pub fn new(limit: usize) -> Self {
        Self {
            undo: Vec::new(),
            redo: Vec::new(),
            limit: limit.max(1),
        }
    }

    /// Records the state *before* an action.
    pub fn push(&mut self, before: S, label: impl Into<String>) {
        self.undo.push((before, label.into()));
        self.redo.clear();
        if self.undo.len() > self.limit {
            self.undo.remove(0);
        }
    }

    /// Returns the previous state and the label of the undone action.
    pub fn undo(&mut self, current: S) -> Result<(S, String), S> {
        match self.undo.pop() {
            Some((prev, label)) => {
                self.redo.push((current, label.clone()));
                Ok((prev, label))
            }
            None => Err(current),
        }
    }

    pub fn redo(&mut self, current: S) -> Result<(S, String), S> {
        match self.redo.pop() {
            Some((next, label)) => {
                self.undo.push((current, label.clone()));
                Ok((next, label))
            }
            None => Err(current),
        }
    }

    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }

    pub fn undo_len(&self) -> usize {
        self.undo.len()
    }

    pub fn redo_len(&self) -> usize {
        self.redo.len()
    }

    /// Labels of all steps, oldest first: the done steps, then the undone
    /// (redoable) ones. The current state sits after `undo_len()` entries.
    pub fn labels(&self) -> Vec<&str> {
        self.undo
            .iter()
            .map(|(_, l)| l.as_str())
            .chain(self.redo.iter().rev().map(|(_, l)| l.as_str()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undo_redo_and_branching() {
        let mut h = History::new(10);
        let mut s = 0;
        for i in 1..=3 {
            h.push(s, format!("set {i}"));
            s = i;
        }
        let (prev, label) = h.undo(s).unwrap();
        assert_eq!((prev, label.as_str()), (2, "set 3"));
        s = prev;
        let (next, _) = h.redo(s).unwrap();
        assert_eq!(next, 3);
        s = h.undo(next).unwrap().0;
        // A new action discards the redo branch.
        h.push(s, "other");
        assert_eq!(h.redo_len(), 0);
        assert!(h.redo(9).is_err());
    }

    #[test]
    fn limit_drops_oldest() {
        let mut h = History::new(2);
        h.push(1, "a");
        h.push(2, "b");
        h.push(3, "c");
        assert_eq!(h.undo_len(), 2);
        let (s, _) = h.undo(4).unwrap();
        let (s, _) = h.undo(s).unwrap();
        assert_eq!(s, 2);
        assert!(h.undo(s).is_err());
    }
}
