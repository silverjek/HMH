# 모델, 학습 스케줄, 런타임 설정 관리
_base_ = [
    '../../_base_/models/tsn_r50.py', '../../_base_/schedules/sgd_100e.py',
    '../../_base_/default_runtime.py'
]

# dataset 경로 및 파일 설정
dataset_type = 'VideoDataset'   # ⭐️ mmaction/datasets/video_dataset.py 참조
# 각각 train, val 영상 경로
data_root = 'data/kinetics400/videos_train'
data_root_val = 'data/kinetics400/videos_val'
# video_path label 형식의 텍스트 파일 (비디오경로 라벨)
ann_file_train = 'data/kinetics400/kinetics400_train_list_videos.txt'   
ann_file_val = 'data/kinetics400/kinetics400_val_list_videos.txt'

file_client_args = dict(io_backend='disk')

# 훈련 파이프라인
train_pipeline = [
    # 비디오 디코더
    dict(type='DecordInit', **file_client_args),
    # 한 비디오에서 3개의 프레임 샘플링
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=3),
    # 프레임 디코딩
    dict(type='DecordDecode'),
    # 짧은 변 기준 256 리사이징
    dict(type='Resize', scale=(-1, 256)),
    dict(
        # 여러 스케일로 자르고 중앙 crop (데이터 증강)
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1),
    # 224x224 고정
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    # 50% 확률로 좌우 반전
    dict(type='Flip', flip_ratio=0.5),
    # NCHW 형태로 변환
    dict(type='FormatShape', input_format='NCHW'),
    # 모델에 넣기 위한 구조로 packing
    dict(type='PackActionInputs')
]

# validation 파이프라인
# 데이터 증강 X
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=3,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    # 중앙 크롭으로 평가 정교화
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

# test 파이프라인
# 데이터 증강 X
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    # 평가 정교화
    dict(type='TenCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

# train 데이터 로드
train_dataloader = dict(
    # 배치 & 병렬 처리 수
    batch_size=32,
    num_workers=8,
    # worker 프로세서 고정; 속도 향상
    persistent_workers=True,

    # shuffle; True면 학습용, False면 평가용
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))

# validation 데이터 로드
val_dataloader = dict(
    batch_size=32,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        
        # test_mode; true면 라벨 무시하고 평가
        test_mode=True))

# test 데이터 로드
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        # test_mode; true면 라벨 무시하고 평가
        test_mode=True))

# 정확도 계산 evaluator
val_evaluator = dict(type='AccMetric')
test_evaluator = val_evaluator

# 3 에포크마다 모델 저장하고 최대 3개 보관
default_hooks = dict(checkpoint=dict(interval=3, max_keep_ckpts=3))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (32 samples per GPU).
'''
GPU 수에 따라서 우리가 조정해야 할 지?.. enable=True 하면 학습률 자동 조정됨
'''
auto_scale_lr = dict(enable=False, base_batch_size=256)
