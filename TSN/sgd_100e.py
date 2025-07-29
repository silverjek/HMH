#======== 학습 루프 설정 ========
train_cfg = dict(

    # epoch 단위로 학습 진행
    type='EpochBasedTrainLoop', 

    # 총 100 에포크 학습
    max_epochs=100, 

    # 1 에포크부터 validation 시작
    val_begin=1, 

    # 매 에포크마다 validation
    val_interval=1)

#======== val&test 루프 설정 ========
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

#======== learningrate scheduling ========
param_scheduler = [
    dict(

        # 정해진 epoch에서 lr을 감소시키는 방식으로 학습
        type='MultiStepLR',

        #에포크 0에서 시작해서
        begin=0,

        # 100에서 종료 (99까지)
        end=100,

        # lr 변경을 에포크 기준으로 적용
        by_epoch=True,

        # 40, 80 에포크에서 lr 감소
        milestones=[40, 80],

        # 해당 시점마다 lr*0.1
        gamma=0.1)
]

#======== optimizer + gradient clipping ========
optim_wrapper = dict(
    optimizer=dict(
        
        # Stochastic Gradient Descent 사용
        type='SGD', 
        
        # 초기 학습률
        lr=0.01, 
        
        # 이전 gradient 방향을 일정 비율 반영해 더 빠르게 수렴
        momentum=0.9, 

        # L2 정규화
        weight_decay=0.0001),

    # Gradient Clipping; backprop 중 gradient L2 norm이 40 이상이면 잘라냄    
    clip_grad=dict(max_norm=40, norm_type=2))
