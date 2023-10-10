import os

from pydantic import BaseModel, Field

from logging1 import custom_logger as logger


class LLMTrainingParams(BaseModel):
    model: str = Field("Llama2", title="Model name")
    data_path: str = Field("data", title="Data path")
    output_dir: str = Field("Project Name", title="Output directory")
    train_split: str = Field("train", title="Train data config")
    valid_split: str = Field(None, title="Validation data config")
    text_column: str = Field("text", title="Text column")
    lr: float = Field(3e-5, title="Learning rate")
    epochs: int = Field(3, title="Number of training epochs")
    batch_size: int = Field(2, title="Training batch size")
    warmup_ratio: float = Field(0.1, title="Warmup proportion")
    gradient_accumulation: int = Field(1, title="Gradient accumulation steps")
    optimizer: str = Field("adamw_torch", title="Optimizer")
    scheduler: str = Field("linear", title="Scheduler")
    weight_decay: float = Field(0.001, title="Weight decay")
    max_grad_norm: float = Field(1, title="Max gradient norm")
    seed: int = Field(42, title="Seed")
    add_eos_token: bool = Field(True, title="Add EOS token")
    block_size: int = Field(1024, title="Block size")
    use_peft: bool = Field(False, title="Use PEFT")
    lora_r: int = Field(64, title="Lora r")
    lora_alpha: int = Field(128, title="Lora alpha")
    lora_dropout: float = Field(0.05, title="Lora dropout")
    logging_steps: int = Field(200, title="Logging steps")
    evaluation_strategy: str = Field("steps", title="Evaluation strategy")
    save_total_limit: int = Field(3, title="Save total limit")
    save_strategy: str = Field("steps", title="Save strategy")
    auto_find_batch_size: bool = Field(False, title="Auto find batch size")
    fp16: bool = Field(True, title="FP16")
    use_int8: bool = Field(False, title="Use int8")
    model_max_length: int = Field(8192, title="Model max length")
    use_int4: bool = Field(False, title="Use int4")
    trainer: str = Field("default", title="Trainer type")
    target_modules: str = Field(None, title="Target modules")
    merge_adapter: bool = Field(False, title="Merge adapter")
    peft_path: str = Field(None,title="adapter model path")
    use_flash_attention_2: bool = Field(True, title="Use flash attention 2")

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "training_params.json")
        # save formatted json
        with open(path, "w") as f:
            f.write(self.json(indent=4))

    def __str__(self):
        data = self.dict()
        data["token"] = "*****" if data.get("token") else None
        return str(data)

    def __init__(self, **data):
        super().__init__(**data)

        # Parameters not supplied by the user
        defaults = {f.name for f in self.__fields__.values() if f.default == self.__dict__[f.name]}
        supplied = set(data.keys())
        not_supplied = defaults - supplied
        if not_supplied:
            logger.warning(f"Parameters not supplied by user and set to default: {', '.join(not_supplied)}")

        # Parameters that were supplied but not used
        # This is a naive implementation. It might catch some internal Pydantic params.
        unused = supplied - set(self.__fields__)
        if unused:
            logger.warning(f"Parameters supplied but not used: {', '.join(unused)}")
