
import argparse
import os
import logging
from typing import Dict, List, Callable, Tuple, Union, Any
import wandb
import uuid
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quant_exp_bias.commands.quantify_exposure_bias_pretrained import quantify_exposure_bias_pretrained

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def get_accum_results_dhawal(run_metrics: Dict[str, Any], prefix='exp_bias'):
    for key in ["disagreements_till_len", "disagreements_at_len"]:
        for key_ in (key, f"target_{key}"):
            if key_ not in run_metrics:
                continue

            for k, values in run_metrics[key_].items():
                for idx, val in values:
                    yield {
                        f'{prefix}/{key_}@{k}/{key}': val,
                        f'len': idx,
                    }
    # list_dicts = []
    # for key in ["kl_till_len", "Hmo_till_len", "Hmm_till_len", "cross_ent_till_len",
    #             "excess_acc_err_till_len", "acc_err_till_len", "kl_till_len_norm",
    #             "model_xent_till_len", "oracle_xent_till_len",
    #             "model_xent_ratio", "target_xent_ratio_norm",
    #             "model_xent_till_len_accum", "oracle_xent_till_len_accum",
    #             "kl_at_len", "Hmo_at_len", "Hmm_at_len", "cross_ent_at_len", 
    #             "model_xent_at_len", "oracle_xent_at_len"]:
    #     for key_ in (key, f"target_{key}", f"ratio_{key}"):
    #         if key_ not in run_metrics:
    #             continue
                
            
    #         for idx, val in run_metrics[key_]:

                
def get_eb_accum_results(run_metrics: Dict[str, Any], prefix='exp_bias'):
    for key in ["kl_till_len", "Hmo_till_len", "Hmm_till_len", "cross_ent_till_len",
                "excess_acc_err_till_len", "acc_err_till_len", "kl_till_len_norm",
                "model_xent_till_len", "oracle_xent_till_len",
                "model_xent_ratio", "target_xent_ratio_norm",
                "model_xent_till_len_accum", "oracle_xent_till_len_accum",
                "kl_at_len", "Hmo_at_len", "Hmm_at_len", "cross_ent_at_len", 
                "model_xent_at_len", "oracle_xent_at_len"]:
        for key_ in (key, f"target_{key}", f"ratio_{key}"):
            if key_ not in run_metrics:
                continue
                
            for idx, val in run_metrics[key_]:
                yield {
                    f'{prefix}/{key_}/{key}': val,
                    f'len': idx,
                }
    
    for key in ["disagreements_till_len", "disagreements_at_len"]:
        for key_ in (key, f"target_{key}"):
            if key_ not in run_metrics:
                continue

            for k, values in run_metrics[key_].items():
                for idx, val in values:
                    yield {
                        f'{prefix}/{key_}@{k}/{key}': val,
                        f'len': idx,
                    }

def get_mean_std_results(run_metrics: Dict[str, Any],
                         prefix:str ='exp_bias', 
                         extras:Dict[str, Any]={}):

    target_model_xent_till_len = run_metrics['target_model_xent_till_len']
    target_kl_till_len_norm = run_metrics['target_kl_till_len_norm']
    kl_till_len_norm = run_metrics['kl_till_len_norm']

    target_xent_data = [(x,y) for ((_, x),(_,y)) in zip(target_model_xent_till_len, target_kl_till_len_norm)]
    
    target_xent_table = wandb.Table(data=target_xent_data,  columns = ["target_xent", "target_kl_till_len"])
    scatter_plot_target = wandb.plot.scatter(target_xent_table, "target_xent", "target_kl_till_len")

    model_xent_data = [(x,y) for ((_, x),(_,y)) in zip(target_model_xent_till_len, kl_till_len_norm)]

    model_xent_table = wandb.Table(data=model_xent_data,  columns = ["target_xent", "model_kl_till_len"])
    scatter_plot_model = wandb.plot.scatter(model_xent_table, "target_xent", "model_kl_till_len")

    output_dict =  {
        f'{prefix}/kl': run_metrics['kl'],
        f'{prefix}/target_kl': run_metrics['target_kl'],
        f'{prefix}/excess_err': run_metrics["excess_err"],
        f'{prefix}/corr_target_err_xent': run_metrics['corr_target_err_xent'],
        f'{prefix}/corr_model_err_xent': run_metrics['corr_model_err_xent'],
        f'{prefix}/scatter_plot_target': scatter_plot_target,
        f'{prefix}/scatter_plot_model': scatter_plot_model,

        f'{prefix}/H_m_m': run_metrics['Hmm'],
        f'{prefix}/H_m_o': run_metrics['Hmo'],
        f'{prefix}/target_H_m_m': run_metrics['target_Hmm'],
        f'{prefix}/target_H_m_o': run_metrics['target_Hmo'],


        f"{prefix}/human_oracle_nll": run_metrics["human_oracle_nll"],
        f"{prefix}/oracle_nll": run_metrics["oracle_nll"],
        f"{prefix}/seq-rep-1": run_metrics["seq-rep-1"],
        f"{prefix}/seq-rep-4": run_metrics["seq-rep-4"],
        f"{prefix}/human-seq-rep-1": run_metrics["human-seq-rep-1"],
        f"{prefix}/human-seq-rep-4": run_metrics["human-seq-rep-4"],
        f"{prefix}/uniq-seq": run_metrics["uniq-seq"],
        f"{prefix}/human-uniq-seq": run_metrics["human-uniq-seq"],

        f"{prefix}/rep/16": run_metrics["rep/16"],
        f"{prefix}/rep/32": run_metrics["rep/32"],
        f"{prefix}/rep/128": run_metrics["rep/128"],
        f"{prefix}/rep/512": run_metrics["rep/512"],
        f"{prefix}/wrep/16": run_metrics["wrep/16"],
        f"{prefix}/wrep/32": run_metrics["wrep/32"],
        f"{prefix}/wrep/128": run_metrics["wrep/128"],
        f"{prefix}/wrep/512": run_metrics["wrep/512"],
        f"{prefix}/hrep/16": run_metrics["hrep/16"],
        f"{prefix}/hrep/32": run_metrics["hrep/32"],
        f"{prefix}/hrep/128": run_metrics["hrep/128"],
        f"{prefix}/hrep/512": run_metrics["hrep/512"],
        f"{prefix}/uniq": run_metrics["uniq"],
        f"{prefix}/human-uniq": run_metrics["human-uniq"],
    }
    extras_ = {}
    for key, val in extras.items():
        extras_[f'{prefix}/{key}'] = val
    output_dict.update(extras_)
    return output_dict

def run_exposure_bias_experiment(experiment_name,
                          config,
                          base_output_dir: str,
                          eval_model:str = 'gpt2',
                          oracle_model:str = 'gpt2-xl',
                          context_file_or_filename: str = 'wikitext-103',
                          split='train',
                          context_len: int = 50,
                          num_samples: int = 100,
                          cuda_device: int = -1,
                          top_k: int = None,
                          top_p: float = None,
                          repeat_penalty: float = None, 
                          beam: int = None,
                          sample_outputs: bool = False,
                          sampling_temperature: float = None,
                          exp_temperature: float = 1.0,
                          generation_size: int = 512,
                          batch_size: int = 16, 
                          experiment_suffix=''):
    output_dir = os.path.join(base_output_dir, 
                    'exp_bias', experiment_suffix)

    experiment = wandb.init(
            dir=output_dir,
            config=config,
            project='qeb',
            group='original')


    metrics = quantify_exposure_bias_pretrained(output_dir,
                eval_model=eval_model, 
                oracle_model=oracle_model,
                context_file_or_filename=context_file_or_filename,
                context_len=context_len,
                num_samples=num_samples,
                cuda_device=cuda_device,
                top_k=top_k,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                beam=beam,
                split=split,
                sample_outputs=sample_outputs,
                sampling_temperature=sampling_temperature,
                exp_temperature=exp_temperature,
                generation_size=generation_size,
                batch_size=batch_size)

    prefix=f'exp_bias/{experiment_suffix}'
    for i in range(generation_size):
        step_data = {}
        for key in ["kl_till_len", "Hmo_till_len", "Hmm_till_len", "cross_ent_till_len",
                    "excess_acc_err_till_len", "acc_err_till_len", "kl_till_len_norm",
                    "model_xent_till_len", "oracle_xent_till_len",
                    "model_xent_ratio", "target_xent_ratio_norm",
                    "model_xent_till_len_accum", "oracle_xent_till_len_accum",
                    "kl_at_len", "Hmo_at_len", "Hmm_at_len", "cross_ent_at_len", 
                    "model_xent_at_len", "oracle_xent_at_len"]:
            for key_ in (key, f"target_{key}", f"ratio_{key}"):
                if key_ not in metrics:
                    continue
                
                if i < len(metrics[key_]):
                    step_data[key_] = metrics[key_][i][1]

       
        experiment.log(step_data , step =i)


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description=f'Exposure Bias Pretrained Experiments.')
  # parser.add_argument('--experiment-name', type=str, 
  #                         required=True,
  #                         help='What experiment is being done.')

  parser.add_argument('--num-samples', type=int, default=10, help='Number of dataset samples to run this iteration for.')

  parser.add_argument('--output-dir', '-o', type=str, default=os.environ.get("OUTPUT_DIR") or os.path.expanduser('~/scratch/quant_exp_bias/'), help='Output directory.')

  parser.add_argument('--oracle-model', type=str, 
                          # default='/home/mila/a/arorakus/wdir/quant_exp_bias/data/gpt2_wikitext103-512',
                          # default='/home/mila/a/arorakus/wdir//quant_exp_bias/data/gpt2_wikitext2-128',
                          # default="/home/mila/a/arorakus/wdir/quant_exp_bias/data/gpt2_orig_wikitext103-512/",
                        #   default='./oracle/',
                        default='./gpt2_models/xl/',
                          # default='gpt2-xl',
                          help='Oracle model.')

  parser.add_argument('--eval-model', type=str, 
                          # default='/home/mila/a/arorakus/wdir//quant_exp_bias/data/gpt2_wikitext2-128',
                          # default='gpt2',
                        #   default='./gpt2_small_wiki/',
                        default='./gpt2_models/small/',
                          help='Oracle model.')

  parser.add_argument('--sample-outputs', action='store_true', 
                      help='Recover the run.')
  
  parser.add_argument('--context-dataset', type=str, 
                      default='wikitext-2',
                      help='Recover the run.')

  parser.add_argument('--context-len', type=int, 
                          default=50,
                          help='Oracle model.')

  cuda_device = parser.add_mutually_exclusive_group(required=False)
  cuda_device.add_argument('--cuda-device',
                            type=int,
                            default=0,
                            help='id of GPU to use (if any)')
  args = parser.parse_args()

  experiment_suffix = 'base/'
  experiment_name = f'qeb_{args.eval_model}_{args.oracle_model}_{str(uuid.uuid4().fields[0])}'

  if args.sample_outputs:
    experiment_name += "_sampled"
  
  run_exposure_bias_experiment(experiment_name, args,
          num_samples=args.num_samples,
          base_output_dir=args.output_dir,
          eval_model=args.eval_model,
          oracle_model=args.oracle_model,
          context_file_or_filename=args.context_dataset,
          context_len=args.context_len,
          cuda_device=args.cuda_device,
          sample_outputs=False,
          experiment_suffix=experiment_suffix,
          top_k = 100,
          top_p = 0.94,
          repeat_penalty = 1.0,
          beam = 5,
          sampling_temperature = 1.0,
          generation_size = 640)
  