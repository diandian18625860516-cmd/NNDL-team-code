# 代码来源与许可

## SALAD

- 作者：Matic Fučka、Vitjan Zavrtanik、Danijel Skočaj。
- 论文：SALAD — Semantics-Aware Logical Anomaly Detection，ICCV 2025。
- 作者仓库：https://github.com/MaticFuc/SALAD
- 代码位于 `vendor/salad/`，导入自本机已有 `SALAD-main` 下载副本；保留 `LICENSE`、README、全部 Python / Shell 源文件和 README 所用的三张方法/结果示意图，省略模型权重。
- 上游提交号：**尚未核实**。准备仓库时无法从终端连接 GitHub，不能把本地副本声称为当前最新提交或已对照上游验证的版本。
- `vendor/SALAD_SOURCE_MANIFEST.json` 记录每个导入文件的 SHA-256；它能检查此副本是否被改动，但不能证明下载来源真实或等同某个上游提交。
- `vendor/salad/LICENSE` 为作者 MIT 许可。源码中的第三方署名和许可注释也保留。论文引用不等于代码由本小组原创。
- 其中 `composition_map_creation/dino/` 文件头注明 Facebook 的 Apache 2.0 许可；本仓库另附标准许可全文 `licenses/Apache-2.0.txt`。

## PatchCore

- 论文：Towards Total Recall in Industrial Anomaly Detection，CVPR 2022。
- 作者仓库：https://github.com/amazon-science/patchcore-inspection
- 本仓库目前只有接入说明，没有复制 PatchCore 实现，也没有核实旧实现与某一上游提交的对应关系。接入时应保存上游提交号、完整许可和实际改动。

## 本小组新增文件

`scripts/`、`tests/`、文档和配置样例为本次整理新增，数据配对检查思路复用之前的本地工作。新增文件的对外开源许可尚未由小组决定，当前按私有课程协作仓库准备；第三方原有许可继续适用。

## 数据与结果

MVTec LOCO AD 来自 https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad ，使用与引用按数据集官网要求执行。数据、权重和历史报告不随代码包分发。作者 README 的结果图属于作者结果，不是本小组成绩。
