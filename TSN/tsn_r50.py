# 전체 모델 구성
model = dict(
    # Recognizer2D: 2D CNN 기반 영상 인식기
    # 📌 mmaction/models/recognizers/recognizer2d.py
    type='Recognizer2D',

    #======== frame-level feature 추출 ========
    # backbone에 각 프레임(혹은 스택된 프레임)을 인풋으로 받아서 피처를 추출
    backbone=dict(

        # ResNet 사용
        # 📌 mmaction/models/backbones/resnet.py 참조
        type='ResNet',

        # ImageNet에서 사전학습된 ResNet-50 가중치 로드
        pretrained='https://download.pytorch.org/models/resnet50-11ad3fa6.pth',

        # ResNet50 사용
        depth=50,

        # 학습 중 BatchNorm의 평균과 분산을 업데이트
        norm_eval=False),
    
    #======== classification 헤더 ========
    # 백본에서 추출한 feature에 대해 spatial average > dropout > FC > softmax 계산
    # frame-level prediction들을 AvgConsensus로 종합
    cls_head=dict(

        # TSN에서 정의된 fully connected + consensus fusion 헤드
        # 📌 mmaction/models/heads/tsn_head.py 참조
        type='TSNHead',

        # Kinetics400 기준 클래스 400개 설정
        num_classes=400,

        # ResNet50의 마지막 feature map 차원 > 이게 FC의 인풋이 됨
        in_channels=2048,

        # ResNet의 출력이 3차원 (C, H, W)
        # spatial average pooling을 수행; 3D 피처맵에 Global Average Pooling을 사용해 벡터로 줄임
        spatial_type='avg',

        # 여러 프레임에서 나온 softmax 확률을 dim=1 기준으로 평균 냄
        consensus=dict(type='AvgConsensus', dim=1),

        # FC 이전에 dropout 적용 (과적합 방지); 40% 확률로 일부 뉴런을 비활성화
        dropout_ratio=0.4,

        # FC 레이어의 weight 초기화
        init_std=0.01,

        # 테스트 시 여러 클립의 예측 확률을 평균
        average_clips='prob'),

    #======== 영상 전처리 설정 ========
    data_preprocessor=dict(

        # mmaction2에서 정의한 표준 비디오 입력 전처리기
        # 📌 mmaction/models/data_preprocessors/data_preprocessor.py 참조
        type='ActionDataPreprocessor',

        # RGB 채널 별 평균값 (ImageNet 통계 기반) - 정규화에 사용
        mean=[123.675, 116.28, 103.53],

        # RGB 채널 별 표춘편차 (ImageNet 통계 기반) - 정규화에 사용
        std=[58.395, 57.12, 57.375],

        # 최종 텐서 shape (N, Channel, Height, Width) 변환 (N개 프레임)
        format_shape='NCHW'),

    # 추가 학습/테스트 전략 설정 - TSN 해당 없음
    train_cfg=None,
    test_cfg=None)
