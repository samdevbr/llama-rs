use std::ffi::CString;

use crate::{
    batch::Batch,
    error::{Error, Result},
};
use llama_sys::{
    llama_backend_free, llama_backend_init, llama_context, llama_context_params, llama_decode,
    llama_free, llama_free_model, llama_load_model_from_file, llama_model, llama_model_params,
    llama_new_context_with_model, llama_token, llama_tokenize,
};

use rand::Rng;

pub const LLAMA_MAX_TENSOR_SPLIT: usize = 16;

pub struct ModelParamsBuilder {
    n_gpu_layers: i32,
    main_gpu: i32,
    tensor_split: Option<[f32; LLAMA_MAX_TENSOR_SPLIT]>,
    vocab_only: bool,
    use_mmap: bool,
    use_mlock: bool,
}

impl Default for ModelParamsBuilder {
    fn default() -> Self {
        Self {
            n_gpu_layers: 0,
            main_gpu: 0,
            tensor_split: None,
            vocab_only: false,
            use_mmap: true,
            use_mlock: false,
        }
    }
}

impl ModelParamsBuilder {
    pub fn tensor_split(mut self, tensor_split: &[f32; LLAMA_MAX_TENSOR_SPLIT]) -> Self {
        self.tensor_split = Some(*tensor_split);
        self
    }

    pub fn vocab_only(mut self, vocab_only: bool) -> Self {
        self.vocab_only = vocab_only;
        self
    }

    pub fn use_mmap(mut self, use_mmap: bool) -> Self {
        self.use_mmap = use_mmap;
        self
    }

    pub fn main_gpu(mut self, main_gpu: i32) -> Self {
        self.main_gpu = main_gpu;
        self
    }

    pub fn n_gpu_layers(mut self, n_gpu_layers: i32) -> Self {
        self.n_gpu_layers = n_gpu_layers;
        self
    }

    pub fn use_mlock(mut self, use_mlock: bool) -> Self {
        self.use_mlock = use_mlock;
        self
    }

    pub fn build(self) -> llama_model_params {
        llama_model_params {
            n_gpu_layers: self.n_gpu_layers,
            main_gpu: self.main_gpu,
            tensor_split: match self.tensor_split {
                Some(t) => t.as_ptr(),
                None => std::ptr::null(),
            },
            progress_callback: None,
            progress_callback_user_data: std::ptr::null::<i32>() as *mut _,
            vocab_only: self.vocab_only,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
        }
    }
}

#[derive(Debug)]
pub struct Model<'a>(&'a llama_model);

impl<'a> Model<'a> {
    pub fn from_file(p: &str, numa: bool, params: llama_model_params) -> Result<Self> {
        let p = CString::new(p).expect("convert path to c-str");

        let m = unsafe {
            let path = p.as_ptr();

            llama_backend_init(numa);

            match llama_load_model_from_file(path, params).as_ref() {
                Some(m) => m,
                None => return Err(Error::ModelCreationFailed),
            }
        };

        Ok(Self(m))
    }

    pub fn tokenize(&self, prompt: String, add_bos: bool, special: bool) -> Vec<llama_token> {
        let mut tokens = Vec::<llama_token>::with_capacity(match add_bos {
            true => prompt.len() + 1,
            false => prompt.len(),
        });

        let prompt = CString::new(prompt).unwrap();

        unsafe {
            let n = llama_tokenize(
                self.as_ref(),
                prompt.as_ptr(),
                prompt.as_bytes().len() as i32,
                tokens.as_mut_ptr(),
                tokens.capacity() as i32,
                add_bos,
                special,
            );

            tokens.set_len(n as usize);
        }

        tokens
    }
}

impl<'a> AsRef<llama_model> for Model<'a> {
    fn as_ref(&self) -> &llama_model {
        self.0
    }
}

impl<'a> Drop for Model<'a> {
    fn drop(&mut self) {
        unsafe {
            llama_free_model(self.0 as *const _ as *mut _);
            llama_backend_free();
        }
    }
}

#[repr(i8)]
pub enum RopeScalingType {
    Unspecified = -1,
    None = 0,
    Linear = 1,
    Yarn = 2,
}

pub struct ContextParamsBuilder {
    seed: u32,
    n_ctx: u32,
    n_batch: u32,
    n_threads: u32,
    n_threads_batch: u32,
    rope_scaling_type: RopeScalingType,
    rope_freq_base: f32,
    rope_freq_scale: f32,
    yarn_ext_factor: f32,
    yarn_attn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_orig_ctx: u32,
    mul_mat_q: bool,
    f16_kv: bool,
    logits_all: bool,
    embedding: bool,
}

impl Default for ContextParamsBuilder {
    fn default() -> Self {
        let mut rng = rand::thread_rng();

        let n_cpu = num_cpus::get() as u32;

        Self {
            seed: rng.gen_range(0..=u32::MAX),
            n_ctx: 512,
            n_batch: 512,
            n_threads: n_cpu,
            n_threads_batch: n_cpu,
            rope_scaling_type: RopeScalingType::Unspecified,
            rope_freq_base: 0.0,
            rope_freq_scale: 0.0,
            yarn_ext_factor: -1.0,
            yarn_attn_factor: 1.0,
            yarn_beta_fast: 32.0,
            yarn_beta_slow: 1.0,
            yarn_orig_ctx: 0,
            mul_mat_q: true,
            f16_kv: true,
            logits_all: false,
            embedding: false,
        }
    }
}

impl ContextParamsBuilder {
    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    pub fn n_ctx(mut self, n_ctx: u32) -> Self {
        self.n_ctx = n_ctx;
        self
    }

    pub fn n_batch(mut self, n_batch: u32) -> Self {
        self.n_batch = n_batch;
        self
    }

    pub fn rope_scaling_type(mut self, rope_scaling_type: RopeScalingType) -> Self {
        self.rope_scaling_type = rope_scaling_type;
        self
    }

    pub fn rope_freq_base(mut self, rope_freq_base: f32) -> Self {
        self.rope_freq_base = rope_freq_base;
        self
    }

    pub fn rope_freq_scale(mut self, rope_freq_scale: f32) -> Self {
        self.rope_freq_scale = rope_freq_scale;
        self
    }

    pub fn yarn_ext_factor(mut self, yarn_ext_factor: f32) -> Self {
        self.yarn_ext_factor = yarn_ext_factor;
        self
    }

    pub fn yarn_attn_factor(mut self, yarn_attn_factor: f32) -> Self {
        self.yarn_attn_factor = yarn_attn_factor;
        self
    }

    pub fn yarn_beta_fast(mut self, yarn_beta_fast: f32) -> Self {
        self.yarn_beta_fast = yarn_beta_fast;
        self
    }

    pub fn yarn_beta_slow(mut self, yarn_beta_slow: f32) -> Self {
        self.yarn_beta_slow = yarn_beta_slow;
        self
    }

    pub fn yarn_orig_ctx(mut self, yarn_orig_ctx: u32) -> Self {
        self.yarn_orig_ctx = yarn_orig_ctx;
        self
    }

    pub fn mul_mat_q(mut self, mul_mat_q: bool) -> Self {
        self.mul_mat_q = mul_mat_q;
        self
    }

    pub fn f16_kv(mut self, f16_kv: bool) -> Self {
        self.f16_kv = f16_kv;
        self
    }

    pub fn logits_all(mut self, logits_all: bool) -> Self {
        self.logits_all = logits_all;
        self
    }

    pub fn embedding(mut self, embedding: bool) -> Self {
        self.embedding = embedding;
        self
    }

    pub fn build(self) -> llama_context_params {
        llama_context_params {
            seed: self.seed,
            n_ctx: self.n_ctx,
            n_batch: self.n_batch,
            n_threads: self.n_threads,
            n_threads_batch: self.n_threads_batch,
            rope_scaling_type: self.rope_scaling_type as i8,
            rope_freq_base: self.rope_freq_base,
            rope_freq_scale: self.rope_freq_scale,
            yarn_ext_factor: self.yarn_ext_factor,
            yarn_attn_factor: self.yarn_attn_factor,
            yarn_beta_fast: self.yarn_beta_fast,
            yarn_beta_slow: self.yarn_beta_slow,
            yarn_orig_ctx: self.yarn_orig_ctx,
            mul_mat_q: self.mul_mat_q,
            f16_kv: self.f16_kv,
            logits_all: self.logits_all,
            embedding: self.embedding,
        }
    }
}

#[derive(Debug)]
pub struct Context<'a>(&'a llama_context);

impl<'a> Context<'a> {
    pub fn with_model<M: AsRef<llama_model>>(m: &M, params: llama_context_params) -> Result<Self> {
        let ctx = unsafe {
            match llama_new_context_with_model(m.as_ref() as *const _ as *mut _, params).as_ref() {
                Some(c) => c,
                None => return Err(Error::ContextCreationFailed),
            }
        };

        Ok(Self(ctx))
    }

    pub fn decode(&self, b: Batch) {
        let ret = unsafe { llama_decode(self.as_ref() as *const _ as *mut _, b.inner) };

        dbg!(ret);
    }
}

impl<'a> AsRef<llama_context> for Context<'a> {
    fn as_ref(&self) -> &llama_context {
        self.0
    }
}

impl<'a> Drop for Context<'a> {
    fn drop(&mut self) {
        unsafe { llama_free(self.0 as *const _ as *mut _) }
    }
}
