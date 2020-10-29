# personality_prediction
demo of  personality_prediction


      modules包的attention.py文件里的第55行，这俩张量（data和weight）的shape最后一维不一样报错了
      
      
              result = data * weight
        # [28, 272, 55, 200] * [28, 272, 55, 200]


运行时请从main.py开始运行
