use llama_rs::{
    error::Result,
    model::{Context, ContextParamsBuilder, Model, ModelParamsBuilder},
};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).unwrap();

    let m_params = ModelParamsBuilder::default().n_gpu_layers(28);
    let m = Model::from_file(&path, false, m_params.build())?;

    let ctx_params = ContextParamsBuilder::default();
    let ctx = Context::with_model(&m, ctx_params.build())?;

    dbg!(&m);
    dbg!(&ctx);

    Ok(())
}
