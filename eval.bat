REM '4 tiny version, disable all'
python main.py ^
    --work test ^
    --cfg ./cfgs/mff_t.yaml ^
    --mlla-disable ^
    --fft-disable ^
    --gate-disable ^
    --data-root ../../datasets/segment ^
    --resume ./logs/lightning_logs/version_4/checkpoints/model_AWSD_100000.ckpt ^
    --batch-size 8 ^
    --grad_freq 1

REM '5 tiny version, enable mlla'
python main.py ^
    --work test ^
    --cfg ./cfgs/mff_t.yaml ^
    --fft-disable ^
    --gate-disable ^
    --data-root ../../datasets/segment ^
    --resume ./logs/lightning_logs/version_5/checkpoints/model_AWSD_140000.ckpt ^
    --batch-size 8 ^
    --grad_freq 1
    
REM '6 tiny version, enable mlla, fft'
python main.py ^
    --work test ^
    --cfg ./cfgs/mff_t.yaml ^
    --gate-disable ^
    --data-root ../../datasets/segment ^
    --resume ./logs/lightning_logs/version_6/checkpoints/last.ckpt ^
    --batch-size 8 ^
    --grad_freq 1 ^
