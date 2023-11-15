use llama_rs::{
    batch::Batch,
    error::Result,
    model::{Context, ContextParamsBuilder, Model, ModelParamsBuilder},
};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).unwrap();

    let m_params = ModelParamsBuilder::default().n_gpu_layers(28);
    let m = Model::from_file(&path, false, m_params.build())?;

    let ctx_params = ContextParamsBuilder::default().seed(1234).n_ctx(2048);
    let ctx = Context::with_model(&m, ctx_params.build())?;

    let prompt = "Hello my name is";
    let tokens = m.tokenize(prompt.to_owned(), false, false);

    for t in &tokens {
        let piece = ctx.token_to_piece(t)?;
        print!("{piece}");
    }

    let mut batch = Batch::new(512, 0, 1);

    for (i, token) in tokens.iter().enumerate() {
        batch.add(*token, i, vec![0], i == tokens.len() - 1);
    }

    ctx.decode(batch)?;

    Ok(())
}
