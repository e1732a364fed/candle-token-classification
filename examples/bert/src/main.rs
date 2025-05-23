use std::path::PathBuf;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use candle_core::{
    utils::{cuda_is_available, metal_is_available},
    Device,
};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::DTYPE;
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use candle_token_classification::{
    BertLikeTokenClassificationHead, BertTokenClassificationHead, Config,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// Use the pytorch weights rather than the safetensors ones
    #[arg(long)]
    use_pth: bool,

    #[arg(long, default_value = "<body>porn</body>")]
    input: String,
}

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

impl Args {
    fn build_model_and_tokenizer(
        &self,
    ) -> Result<(BertTokenClassificationHead, Tokenizer, Vec<String>)> {
        let device = device(self.cpu)?;
        let default_model = "google-bert/bert-base-multilingual-cased".to_string();
        let default_revision = "main".to_string();
        let (model_id, revision) = match (self.model.to_owned(), self.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id.clone(), RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            if model_id.starts_with("/") {
                (
                    PathBuf::from(model_id.clone() + "/config.json"),
                    PathBuf::from(model_id.clone() + "/tokenizer.json"),
                    PathBuf::from(model_id + "/model.safetensors"),
                )
            } else {
                let api = Api::new()?;
                let api = api.repo(repo);
                let config = api.get("config.json")?;
                let tokenizer = api.get("tokenizer.json")?;
                let weights = if self.use_pth {
                    api.get("pytorch_model.bin")?
                } else {
                    api.get("model.safetensors")?
                };
                (config, tokenizer, weights)
            }
        };
        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(&tokenizer_filename).map_err(E::msg)?;
        let labels = config.id2label.values().cloned().collect();

        let vb = if self.use_pth {
            VarBuilder::from_pth(&weights_filename, DTYPE, &device)?
        } else {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? }
        };
        let model = BertTokenClassificationHead::load(vb, &config)?;

        Ok((model, tokenizer, labels))
    }
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let (model, tokenizer, labels) = args.build_model_and_tokenizer()?;
    let sentence = &args.input;

    println!("classify {sentence} ...");
    let output = model.classify(sentence, &tokenizer, &model.device)?;
    println!("{:?}, {}", labels[output.0 as usize], output.1.to_string());

    Ok(())
}
