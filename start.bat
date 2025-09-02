REM '3 base version'
python main.py ^
    --work fit ^
    --cfg ./cfgs/mff_b.yaml ^
    --data-root ../../datasets/segment ^
    --batch-size 4 ^
    --grad_freq 2
    