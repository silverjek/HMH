default_scope = 'mmaction'

#======== 학습 중 hook 설정 ========
default_hooks = dict(
    # GPU memory 사용량, 학습 스탯 등 실시간 정보 출력
    runtime_info=dict(type='RuntimeInfoHook'),
    # 각 iteration의 시간 측정
    timer=dict(type='IterTimerHook'),
    # 로그 출력; 20 iteration마다 loss 등 출력
    logger=dict(type='LoggerHook', interval=20, ignore_last=False),
    # param_scheduler 설정대로 lr 등 스케줄 적용
    param_scheduler=dict(type='ParamSchedulerHook'),
    # epoch마다 모델 저장; 성능 가장 좋은 모델 자동 저장
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto'),
    # 분산 학습 시 seed 고정 (재현성 보장)
    sampler_seed=dict(type='DistSamplerSeedHook'),
    # 분산 학습 시 BN 등 buffer sync 맞춤
    sync_buffers=dict(type='SyncBuffersHook'))

#======== 실행 환경 설정 ========
env_cfg = dict(
    cudnn_benchmark=False,
    # 다중 프로세싱 설정
    mp_cfg=dict(
        # 멀티프로세싱 초기화 방식 (Linux 환경 권장값)
        mp_start_method='fork', 
        # OpenCV 병렬 처리 비활성화
        opencv_num_threads=0),
    # 분산 학습 환경 설정
    dist_cfg=dict(backend='nccl'))  # NVIDIA GPU용 분산 backend (PyTorch 공식 권장)

# log 출력 설정
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)

# 시각화 데이터를 로컬 디렉토리에 저장
vis_backends = [dict(type='LocalVisBackend')]
# 영상 인식 모델의 시각화 도구
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

# 로그 출력의 상세 정도를 조절
log_level = 'INFO'
# 사전학습된 모델 weight를 불러오지 않음 (config에서 직접 preload)
load_from = None
# 이전 학습을 이어서 진행하지 않음 (처음부터 학습 시작)
resume = False
