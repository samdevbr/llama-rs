use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to load the model")]
    ModelCreationFailed,
    #[error("failed to create the context")]
    ContextCreationFailed,
    #[error("failed to decode batch {code}")]
    BatchDecodeFailed { code: i32 },
}

pub type Result<T> = std::result::Result<T, Error>;
