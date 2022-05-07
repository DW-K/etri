# etri
멀티모달 데이터를 통한 사용자 맞춤 감정 분석 source code

# function description

1. integ_ts.integ:
  1) description: lifelog 데이터를 1 day 단위로 합쳐주는 function
  2) input: userNum(str)
  2) output: dataframe(at second memory)

2. model.train
  1) description: model train file
  2) input: batch_size, sequence_size, interval, user_list, target_col, drop_col, num_layers, hidden_size, lr, epochs, model_func, model_name
  3) output: model.pt
  4) input 설명
    a. interval : 센서 데이터의 간격 (ex: 1 -> 1초 마다)
    b. user_list : user number 의 list (ex: ['01', '02'])
    c. model_func : model.model에 정의되어있는 모델  

3. model.predict
  1) description: model predict file
  2) input: model_name, model_func
  3) output: model.pt

4. model.fine_tuning
  1) description: model predict file
  2) input: model_name, epochs, user_num, model_func
  3) output: fine_tuning_model.pt

# 사용 방법
1. integ_ts.py를 실행시키면 데이터를 넣을 폴더가 생성됨 (dataset)
2. data를 넣고 integ_ts.py 다시 실행
3. model.train, model.predict, model.fine_tuning 사용 가능 (main 참고)
