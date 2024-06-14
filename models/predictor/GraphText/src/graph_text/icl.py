from omegaconf import DictConfig

from llm.llm import LLM
from utils.data.textual_graph import TextualGraph


class LLMForInContextLearning(object):
    def __init__(self, cfg: DictConfig, data: TextualGraph, llm: LLM, _logger, max_new_tokens=20, gen_mode="text",
                 **kwargs, ):
        self.cfg = cfg
        self.gen_mode = gen_mode
        self.data = data
        self.text = data.text
        self.logger = _logger
        self.llm = llm
        self.max_new_tokens = max_new_tokens
        # ! Classification prompt

        self.text["dialog"] = "NA"
        self.text["demo"] = "NA"
        self.text["question"] = "NA"
        self.text["generated_text"] = "NA"

    def eval_and_save(self, step, sample_node_id, split):
        res_df = self.text.dropna()
        res_df["correctness"] = res_df.apply(lambda x: x["gold_choice"] in x["pred_choice"], axis=1)
        res_df.sort_values('correctness', inplace=True)
        save_file = self.cfg.save_file.format(split=split)
        res_df.to_csv(save_file)
        acc = res_df["correctness"].mean()
        self.logger.save_file_to_wandb(save_file, base_path=self.cfg.out_dir)
        valid_choice_rate = (res_df["pred_choice"].isin(self.data.choice_to_label_id.keys()).mean())
        acc_in_valid_choice = acc / valid_choice_rate if valid_choice_rate > 0 else 0
        result = {
            "out_file": save_file,
            f"{split}_acc": acc,
            f"{split}_valid_choice_rate": valid_choice_rate,
            f"{split}_acc_in_valid_choice": acc_in_valid_choice,
        }
        if valid_choice_rate > 0:
            valid_df = res_df[res_df["pred_choice"].isin(self.data.choice_to_label_id.keys())]
            valid_df["true_choices"] = valid_df.apply(lambda x: self.data.label_info.choice[x["label_id"]], axis=1)
            result.update({f"PD/{choice}.{self.data.choice_to_label_name[choice]}": cnt / len(valid_df)
                           for choice, cnt in valid_df.pred_choice.value_counts().to_dict().items()})
        sample = {f"sample_{k}": v
                  for k, v in self.data.text.iloc[sample_node_id].to_dict().items()}
        self.logger.info(sample)
        self.logger.wandb_metric_log({**result, "step": step})

        #  ! Save statistics to results
        # y_true, y_pred = [valid_df.apply(lambda x: self.data.l_choice_to_id[x[col]], axis=1).tolist() for col in
        #                   ('true_choices', 'pred_choice')]
        # result['cla_report'] = classification_report(
        #     y_true, y_pred, output_dict=True,
        #     target_names=self.data.label_info.label_name.tolist())
        self.logger.info(result)
        self.logger.critical(f"Saved results to {save_file}")
        self.logger.wandb_summary_update({**result, **sample})
        return result

    def __call__(self, node_id, prompt, demo, question, log_sample=False):
        # ! Classification
        prompt = prompt + " " if prompt.endswith(":") else prompt  # ! Critical
        if self.gen_mode == "choice":
            generated = self.llm.generate_text(prompt, max_new_tokens=1, choice_only=True)
            pred_choice = generated[-1] if len(generated) > 0 else "NULL"
        else:
            generated = self.llm.generate_text(prompt, self.max_new_tokens)
            try:
                pred_choice = generated.split("<answer>")[-1][0]  # Could be improved
            except:
                pred_choice = ""

        self.text.loc[node_id, "dialog"] = prompt + generated
        self.text.loc[node_id, "demo"] = demo
        self.text.loc[node_id, "question"] = question
        self.text.loc[node_id, "pred_choice"] = pred_choice
        self.text.loc[node_id, "generated_text"] = generated
