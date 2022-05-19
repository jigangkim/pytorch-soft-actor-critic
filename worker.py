import argparse
import datetime
import os.path as osp

from nvidia_gpu_scheduler.scheduler import NVGPUScheduler
from nvidia_gpu_scheduler.worker import NVGPUWorker
from nvidia_gpu_scheduler.utils import CatchExceptions

from main import main


class SACWorker(NVGPUWorker):
    log_basedir = ''

    @staticmethod
    @CatchExceptions
    def worker_function(*args, config_path=None, config=None, config_byte=None, **kwargs):
        import gin
        import os
        from tqdm import tqdm

        from nvidia_gpu_scheduler.utils import log_tqdm

        @gin.configurable
        def run(
            env_name='HalfCheetah-v2',
            policy='Gaussian',
            eval=True,
            gamma=0.99,
            tau=0.005,
            lr=0.0003,
            alpha=0.2,
            automatic_entropy_tuning=False,
            seed=123456,
            batch_size=256,
            num_steps=1000001,
            hidden_size=256,
            updates_per_step=1,
            start_steps=10000,
            target_update_interval=1,
            replay_size=1000000,
            cuda=False
            ):

            # log_tqdm
            pbar = tqdm(total=num_steps)
            log_tqdm(pbar, config_path.replace('/', '_'))
            
            main(
                env_name=env_name,
                policy=policy,
                eval=eval,
                gamma=gamma,
                tau=tau,
                lr=lr,
                alpha=alpha,
                automatic_entropy_tuning=automatic_entropy_tuning,
                seed=seed,
                batch_size=batch_size,
                num_steps=num_steps,
                hidden_size=hidden_size,
                updates_per_step=updates_per_step,
                start_steps=start_steps,
                target_update_interval=target_update_interval,
                replay_size=replay_size,
                cuda=cuda,
                log_basedir=SACWorker.log_basedir,
            )

        tmp_gin_file = '/tmp/%s'%(config_path.replace('/','_'))
        with open(tmp_gin_file,'wb') as f:
            f.write(config_byte)
        gin.parse_config_file(tmp_gin_file)
        os.remove(tmp_gin_file)
        run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    list_of_identities = ['scheduler', 'worker']
    parser.add_argument('--identity', type=str, choices=list_of_identities,
        help='Specify identity. Available identities are %s'%list_of_identities
    )
    parser.add_argument('--port', type=int, default=12345,
        help='Specify port number for scheduler/worker. Default is 12345'
    )
    parser.add_argument('--basedir', type=str, default=osp.dirname(osp.abspath(__file__)),
        help='Specify base directory for configs and runs. Default is workspace directory'
    )
    parser.add_argument('--subdir', type=str, required=True,
        help='Specify subdirectory for configs and runs'
    )
    args = parser.parse_args()

    if args.identity == 'scheduler':
        scheduler = NVGPUScheduler(args.port, 'pytorch-soft-actor-critic')
        scheduler.start()
        scheduler.run(path_to_configs=osp.join(args.basedir, 'configs', args.subdir),
            config_extension='.gin'
        )
    elif args.identity == 'worker':
        SACWorker.log_basedir = osp.join(args.basedir, 'runs', args.subdir)
        worker = SACWorker('127.0.0.1', args.port, 'pytorch-soft-actor-critic')
        worker.connect()
        worker.update_limits(
            available_gpus=[0,1,2,3],
            gpu_utilization_limit=[100,100,100,100],
            gpu_job_limit=[5,5,5,5],
            utilization_margin=0,
            max_gpu_mem_usage=90,
            time_between_jobs=30,
            subprocess_verbose=False,
            apply_limits='worker'
        )
        worker.run()