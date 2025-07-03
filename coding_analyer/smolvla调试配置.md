```json
{
    // 使用 IntelliSense 了解相关属性。 
    // 悬停以查看现有属性的描述。
    // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "smolvla test",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/lerobot/scripts/train.py",
            "console": "integratedTerminal",
            "env": {
                "CUDA_VISIBLE_DEVICES": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
            "args": [
                "--policy.path=/data1/workspace/huqiong/train_log/smolvla_base",
                "--dataset.repo_id=test",
                "--dataset.root=/data1/datasets/can_remove/hq_workspace/test_dataset/ori_data",
                "--batch_size=1",
                "--steps=2000",
                "--output_dir=/data1/workspace/huqiong/train_log/lerobot/smolvla/test",
                "--wandb.enable=true",
                "--wandb.project=smolvla_test",
                "--wandb.mode=offline",
            ],
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "sh_debug",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 52011
            }
        },
        {
            "name": "dataset_unit_test",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/hq_unit_test/dataset_test.py",
            "console": "integratedTerminal",
            "env": {
                "CUDA_VISIBLE_DEVICES": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
            "args": [],
            "cwd": "${workspaceFolder}",
            "justMyCode": false,
        },
        {
            "name": "model_test",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/hq_test/test.py",
            "console": "integratedTerminal",
            "env": {
                "CUDA_VISIBLE_DEVICES": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
            "args": [],
            "cwd": "${workspaceFolder}",
            "justMyCode": false,
        },
        {
            "name": "pi0 test train",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/lerobot/scripts/train.py",
            "console": "integratedTerminal",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "TOKENIZERS_PARALLELISM": "false",
            },
            "args": [
                "--policy.type=pi0",
                // "--policy.path=/data1/model_weight/models--lerobot--pi0/snapshots/8f50aacbe079a026391616cf22453de528f2a873",
                "--config_path=/data1/model_weight/models--lerobot--pi0/snapshots/8f50aacbe079a026391616cf22453de528f2a873",
                // "--policy.train_expert_only=true",
                "--policy.tokenizer_path=/data1/model_weight/models--google--paligemma-3b-pt-224/snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c",
                "--dataset.repo_id=test",
                "--dataset.root=/data1/datasets/can_remove/hq_workspace/test_dataset/ori_data",
                "--batch_size=1",
                "--steps=2000",
                "--output_dir=/data1/workspace/huqiong/train_log/lerobot/pi0/test",
                "--wandb.enable=true",
                "--wandb.project=pi0_test",
                "--wandb.mode=offline",
            ],
            "cwd": "${workspaceFolder}"
        },
    ]
}
```