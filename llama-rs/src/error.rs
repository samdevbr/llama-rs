use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("failed to load the model")]
    ModelCreationFailed,
    #[error("failed to create the context")]
    ContextCreationFailed,
    #[error("failed to decode batch {code}")]
    BatchDecodeFailed { code: i32 },
    #[error("failed to convert token to piece {code}")]
    TokenPieceConvertionFailed { code: i32 },
    #[error("failed to convert byte vector to utf8 string")]
    ByteArrayToStringFailed(#[from] std::string::FromUtf8Error),
}

pub type Result<T> = std::result::Result<T, Error>;
