use llama_sys::{
    llama_batch, llama_batch_free, llama_batch_init, llama_pos, llama_seq_id, llama_token,
};

#[derive(Debug)]
pub struct Batch<'a> {
    pub inner: llama_batch,
    cursor: usize,
    tokens: &'a mut [llama_token],
    positions: &'a mut [llama_pos],
    seq_id_lens: &'a mut [i32],
    seq_ids: Vec<&'a mut [llama_seq_id]>,
    logits: &'a mut [i8],
}

impl<'a> Batch<'a> {
    pub fn new(n_tokens: usize, embd: usize, n_seq_max: usize) -> Self {
        let b = unsafe { llama_batch_init(n_tokens as i32, embd as i32, n_seq_max as i32) };

        let tokens = unsafe { std::slice::from_raw_parts_mut(b.token, n_tokens) };
        let positions = unsafe { std::slice::from_raw_parts_mut(b.pos, n_tokens) };
        let n_seq_id = unsafe { std::slice::from_raw_parts_mut(b.n_seq_id, n_tokens) };
        let logits = unsafe { std::slice::from_raw_parts_mut(b.logits, n_tokens) };
        let seq_id = unsafe { std::slice::from_raw_parts_mut(b.seq_id, n_tokens) };

        let ids = seq_id
            .iter()
            .map(|seq| unsafe { std::slice::from_raw_parts_mut(*seq, n_seq_max) })
            .collect();

        Self {
            tokens,
            logits,
            positions,
            inner: b,
            cursor: 0,
            seq_ids: ids,
            seq_id_lens: n_seq_id,
        }
    }

    pub fn clear(&mut self) {
        self.cursor = 0;
    }

    pub fn add(
        &mut self,
        token: llama_token,
        pos: usize,
        seq_ids: Vec<llama_seq_id>,
        logits: bool,
    ) {
        self.tokens[self.cursor] = token;
        self.positions[self.cursor] = pos as llama_pos;
        self.seq_id_lens[self.cursor] = seq_ids.len() as i32;
        self.logits[self.cursor] = match logits {
            true => 1,
            false => 0,
        };

        for (i, seq) in seq_ids.iter().enumerate() {
            self.seq_ids[self.cursor][i] = *seq;
        }

        self.cursor += 1;
        self.inner.n_tokens = self.cursor as i32;
    }
}

impl<'a> Into<llama_batch> for Batch<'a> {
    fn into(self) -> llama_batch {
        self.inner
    }
}

impl<'a> Drop for Batch<'a> {
    fn drop(&mut self) {
        unsafe { llama_batch_free(self.inner) }
    }
}

#[cfg(test)]
mod tests {
    use super::Batch;

    #[test]
    fn test_batch_values() {
        let batch = Batch::new(512, 0, 1);

        assert_eq!(batch.tokens.len(), 512);

        for (i, token) in batch.tokens.iter_mut().enumerate() {
            let raw_token = unsafe {
                let addr = batch.inner.token.add(i);

                std::ptr::read(addr)
            };

            assert_eq!(token, &raw_token);

            *token = 0;

            let raw_token = unsafe {
                let addr = batch.inner.token.add(i);

                std::ptr::read(addr)
            };

            assert_eq!(0, raw_token);
        }
    }
}
