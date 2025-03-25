# 勇者游戏
# 
# 为了更好复习MC/TD，我特别设计了一个简单的小游戏：
# 在地图上，有town, forest, castle三个地方，代表三个状态
# 勇者（agent）总是出生在town
# 每个回合中，勇者必须进行10次游戏，每次游戏是一个timestep
# 每个timestep中，勇者必须前往另一个场景；比如勇者在town时，他必须选择前往castle或者forest
# 在勇者从一个地方到另一个地方的时候，有可能成功，也有可能失败，即状态转移概率
# 如果转移成功，则获得对应场景的奖励；如果失败，则获得对应场景的惩罚
#
# 各场景的奖励和惩罚分别如下：（Reward）
#    town: +5 / -1
#    castle: +8 / -5
#    forest: +2 / -3
#
# 各场景之间的转移成功率如下： P (S' | S, a)
#    town to castle: 0.1
#    town to forest: 0.7
#    forest to castle: 0.1
#    forest to town: 0.8
#    castle to town: 0.3
#    castle to forest: 0.8
#
# Action：在每个地方，必须选择另外两个地方中的一个作为该回合的action
#
# Policy：如果有具体的策略，则可进行prediction (policy evaluation)