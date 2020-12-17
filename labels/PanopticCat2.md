# Texture-agnostic
# For a class-agnostic detector is often used as a pre-processor: to produce a bunch of interesting bounding boxes that have a high chance of containing cat, dog, car, etc. Obviously, we need a specialized classifier after a class-agnostic detector to actually know what class each bounding box contains.
# Similarly, here we define texture-agnostic detectors that produce a bunch of interesting bounding boxes or instance segments that have a high chance of containing door, chair, table etc. So we need a specialized classifier after a texture-agnostic detector to actually know what texture each object instance consists of.

Thing: (count) 计数/实例分割
    person 人
    table 桌
    chair 椅
    couch 沙发
    wall 墙
    door 门
    door_opening 门洞
    window 窗
    window_opening 窗洞
    floor 地板
    ceiling 天花板
    stairs 楼梯
    handrail 扶手
    sign 标志
    plant 植物
    electronic 电子设备
    appliance 电器
    light 灯
Stuff: (no count) 不计数/语义分割
    sky 天空
    road 公路
    aisle 走廊

Texture：for wall, door, floor, ceiling, table, chair, couch, stairs ...
    wood 木制
    metal 金属
    glass 玻璃
    fabric 织物
    plastic 塑料/塑胶
    leather 皮质
    paint 涂漆
    tile 瓷砖
    brick 砖
    marble 大理石
    concrete 水泥/混凝土
    electronic-screen 电子屏

plant植物：such as tree树, grass草, flower花, bonsai盆栽, ...
sign标识: such as banner横幅, poster海报, doorplate门牌, sign_toilet卫生间牌, sign_safety安全标识, sign_number数字标识, ...
electronic电子设备：such as mobile手机, laptop便携电脑, desktop台式机, mouse鼠标, keyboard键盘, monitor/tv显示器/电视, remote-control遥控器, door-control门禁, alertor警报器, switch(灯/空调/电源)开关, ...
appliance电器：such as kettle电水壶, microwave微波炉, refrigerator冰箱, air-conditioner空调, water-fountain饮水机, coffee-machine咖啡机, copying-machine复印机, vending-machine自动贩卖机, ...