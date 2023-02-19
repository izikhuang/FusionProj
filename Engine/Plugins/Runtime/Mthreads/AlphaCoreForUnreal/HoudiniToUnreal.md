# **如何把AlphaCore在Houdini中解算效果移植到Unreal**

## 1. 资产导出
* 将Houdini的高度场转为模型后，导出FBX文件备用
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/MountainExport.jpg" alt="drawing"/>
    </p>
* 将FBX文件导入到Unreal中，并调整位置。
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/ImportOptions.jpg" alt="drawing"/>
    </p>

    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/ImportUnreal.jpg" alt="drawing"/>
    </p>
* 这里山的位置设置在原点，因为Houdini高度场的位置就在原点

## 2. 设置StormActor的位置及大小
* 创建StormActor
* 设置解算域位置和大小
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/HoudiniFieldSize.jpg" alt="drawing"/>
    </p>
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/ActorTransform.jpg" alt="drawing"/>
    </p>

    Houdini的Pivot为`{0,32,0}`,由于Houdini的坐标系与Unreal有所差异，Unreal中的坐标应乘以100倍，并调换Y/Z坐标。Unreal的坐标应为`{0,0,3200}`。

    Houdini的Size为`{125,64,125}`,Unreal中Actor绑定的是UBoxComponent，默认大小为`{64,64,64}`,所以调整Unreal的Scale大小为`{125*100/64,125*100/64,64*100/64}`，即`{195.3125,195.3125,100}`

    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/HoudiniField.jpg" alt="drawing"/>
    </p>
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/UnrealField.jpg" alt="drawing"/>
    </p>
* Unreal中设置VoxelSize为0.3，与Houdini一致
* 将Mountain Add到StormActor中
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/AddMountain.jpg" alt="drawing"/>
    </p>
## 3. 设置发射器
* 将发射器添加到StormActor中
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/addEmitter.jpg" alt="drawing"/>
    </p>
    由于houdini中只使用了一个发射器，Unreal中也只需添加一个发射器
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/HoudiniAxFieldSource.jpg" alt="drawing"/>
    </p>

* 设置发射器的位置与大小
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/EmitterSize.jpg" alt="drawing"/>
    </p>
    与Actor位置类似，这里需要设置发射器的位置和size如图
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/UnrealEmitterSize.jpg" alt="drawing"/>
    </p>
    Houdini与Unreal位置对比
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/UnrealEmitterView.jpg" alt="drawing"/>
    </p>
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/HoudiniEmitterView.jpg" alt="drawing"/>
    </p>

* 设置发射器其他参数
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/HEmitterParm1.jpg" alt="drawing"/>
    </p>
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/HEmitterParm2.jpg" alt="drawing"/>
    </p>
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/UnrealEmitterParm.jpg" alt="drawing"/>
    </p>

* 设置Actor其他参数
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/HoudiniParms1.jpg" alt="drawing"/>
    </p>
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/HoudiniParms2.jpg" alt="drawing"/>
    </p>
    <p align ="center">
        <img src="./Resources/DocImage/HoudiniToUnreal/UnrealStormParms.jpg" alt="drawing"/>
    </p>

## 4. 效果对比

<p align ="center">
    <img src="./Resources/DocImage/HoudiniToUnreal/Houdini.jpg" alt="drawing"/>
</p>

<p align ="center">
    <img src="./Resources/DocImage/HoudiniToUnreal/Unreal.jpg" alt="drawing"/>
</p>