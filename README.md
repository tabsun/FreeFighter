### ——基于PaddleHub的真人街霸游戏
- [bilibili](https://www.bilibili.com/video/BV1qi4y1P7db)
- [AI studio](https://aistudio.baidu.com/aistudio/projectdetail/1697846)
- [CSDN](https://blog.csdn.net/delltdk/article/details/115145668)

街霸（Street Fighter）是大家非常熟悉的一个游戏。小时候我们都会和小伙伴们互相喊着“阿斗根”来发大招。现在借助于Paddlehub提供的视频人物分析技术，我们可以进入到街霸的世界里，虐别人和被虐。

![](https://img-blog.csdnimg.cn/img_convert/30cf88a011723144128ba34c5fb2bae6.png)

## 1. 游戏展示

b站链接：[https://www.bilibili.com/video/BV1qi4y1P7db/](https://www.bilibili.com/video/BV1qi4y1P7db/)


## 2. 实现思路
![](https://img-blog.csdnimg.cn/img_convert/5d99df7b69181f82ef4e219f6471acb0.png)
在视频中查找与游戏人物动作最接近的frame，抽取其中的人体部分，生成相应的GIF动图，作为游戏人物的素材。运行时左右侧游戏角色分别为images/RYU1和images/RUYU2.

## 3. Requirements

- numpy
- paddlepaddle
- paddlehub
- opencv-python

## 4. RUN
```
python demo.py --search_video mp4/dance.mp4

cp output/*.gif StreetFighter/images/RYU1/

cd StreetFighter
```

浏览器打开StreetFighter/index.html即可，具体操作说明参见StreetFighter/README。

