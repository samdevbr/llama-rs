use llama_sys::{llama_batch, llama_batch_init, llama_token};

#[derive(Debug)]
pub struct Batch {
    b: llama_batch,
}

impl Batch {
    pub fn new(n_tokens: i32, embd: i32, n_seq_max: i32) -> Self {
        let b = unsafe { llama_batch_init(n_tokens, embd, n_seq_max) };

        Self { b }
    }

    pub fn tokens(&self) -> Vec<llama_token> {
        let tokens = unsafe { std::slice::from_raw_parts(self.b.token, self.b.n_tokens as usize) };

        tokens.to_vec()
    }
}
