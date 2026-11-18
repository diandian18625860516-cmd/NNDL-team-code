# 数据和结果约定

MVTec LOCO AD 有五个产品类别，正式目录名是 `screw_bag`、`pushpins`、`breakfast_box`、`juice_bottle`、`splicing_connectors`。官网：https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad 。SALAD 作者组成图与权重入口见作者仓库 README 和下载脚本。

## 路径

`dataset_root` 指向包含五个类别文件夹的目录，`composition_root` 指向包含五个类别组成图文件夹的目录。两者分别有 `类别/train/...`、`类别/validation/...`、`类别/test/...`。

共享的图片标识使用相对于数据集根目录的路径，例如 `screw_bag/train/good/000.png`，不能只记录 `000.png`。原始数据集划分保持不变；训练编辑源图、开发源图、最终诊断源图的用途另用清单记录。

两张图路径匹配只证明能找到对应文件；还要核对能否打开、图像与组成图是否对应、标签数字的真实含义。作者组成图、自制组成图、ground_truth 异常标注是不同材料。

## 原图用途和编辑样本

新用途表要覆盖所有使用到的源图；同一源图不得同时用于训练和最终诊断。归一化、调参以及初始检查点曾使用哪些源图也要记录。最终诊断只在方案冻结后评价。

编辑样本清单至少包括 source_path、合法版本路径、违规版本路径、用途、组成图表示版本、变换参数、规则依据、审核状态和审核人。自动程序通过不等于人工规则审核通过。

## 结果

逐图预测至少包含 `run_id,image_path,anomaly_type,label,score`。label 统一 0 为正常、1 为异常；score 的方向在运行记录说明。每张官方测试图一条对应分支的预测，不漏图、不重复。

若输出组成、外观和融合多个分数，明确使用的列/分支。AUROC 是排序表现，不能叫分类准确率；logical 和 structural 分别与正常图比较，macro 为二者平均，pooled 合并所有异常后计算。

运行记录参考 `templates/run_record.example.json`，null 表示尚未填写，不允许当实际成绩。至少记录代码提交号、权重来源及哈希、数据清单、训练设置、环境和分数含义。作者权重复评、自训练、同预算继续训练分别标明。原图与编辑诊断结果分开报告。

小规模、已经验证且无敏感内容的结果表可放到后续建立的 `reports/` 中提交；原始权重和大输出保存到忽略的 `outputs/` 或共享存储，仓库只记录获取方式和校验值。
