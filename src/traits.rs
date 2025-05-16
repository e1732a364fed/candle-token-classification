use candle_core::{Device, Error as E, Module, Result, Tensor};
use candle_nn::{linear, ops::softmax_last_dim, Dropout, Linear, VarBuilder};
use tokenizers::Tokenizer;

use super::Config;

pub trait BertLikeModel: Sized {
    type Config<'a>: From<&'a Config>;

    fn load<'a>(vb: VarBuilder, config: &Self::Config<'a>) -> Result<Self>;
    fn device(&self) -> &Device;
    fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor>;
}

pub trait BertLikeTokenClassificationHead: Sized {
    type Model: BertLikeModel;

    fn new(
        device: Device,
        model: Self::Model,
        dropout: Dropout,
        classifier: Linear,
        pooler_dense: Linear,
    ) -> Self;
    fn model(&self) -> &Self::Model;
    fn dropout(&self) -> &Dropout;
    fn classifier(&self) -> &Linear;
    fn pooler_dense(&self) -> &Linear;

    fn load(vb: VarBuilder, config: &Config) -> Result<Self> {
        let vs = vb.pp("classifier");
        let model = Self::Model::load(vb.clone(), &config.into())?;
        let num_labels = config.id2label.len();
        let classifier_dropout = config
            .classifier_dropout
            .unwrap_or(config.hidden_dropout_prob);

        let hs = config.hidden_size;

        Ok(Self::new(
            model.device().clone(),
            model,
            Dropout::new(classifier_dropout),
            linear(hs, num_labels, vs)?,
            linear(hs, hs, vb.pp("bert.pooler.dense"))?,
        ))
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let outputs = self
            .model()
            .forward(input_ids, token_type_ids, attention_mask)?;
        use candle_core::IndexOp;

        let first_token_tensor = outputs.i((.., 0))?;
        let pooled_output = self.pooler_dense().forward(&first_token_tensor)?;
        let pooled_output = pooled_output.tanh()?;

        let sequence_output = self.dropout().forward(&pooled_output, false)?;
        let logits = self.classifier().forward(&sequence_output)?;

        Ok(logits)
    }

    fn classify<'a>(
        &self,
        s: &'a str,
        tokenizer: &Tokenizer,
        device: &Device,
    ) -> Result<(u32, Tensor)> {
        let Ok(mut token_encoding) = tokenizer.encode(s, true) else {
            return Err(E::Msg("encoding".to_string()));
        };
        let pad_token = "[PAD]";
        let pad_type_id = 0;
        let pad_id = 0; //这个才是真正填进去的

        token_encoding.pad(
            512,
            pad_id,
            pad_type_id,
            pad_token,
            tokenizers::PaddingDirection::Right,
        );
        let tokens = token_encoding.get_ids().to_vec();
        let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?; //vec转 tensor 后, 再加一维度
        let token_type_ids = input.zeros_like()?;

        let attention_mask = Tensor::new(
            token_encoding.get_attention_mask().to_vec().as_slice(),
            device,
        )?
        .unsqueeze(0)?;

        let logits = self
            .forward(&input, &token_type_ids, Some(&attention_mask))?
            .squeeze(0)?;
        let scores = softmax_last_dim(&logits)?;

        let label_indices = scores.argmax(0)?;
        let v = label_indices.to_vec0::<u32>()?;
        Ok((v, scores))
    }
}
