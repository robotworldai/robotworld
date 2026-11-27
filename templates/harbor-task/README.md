# Harbor task authoring scaffold / Harbor 造题模板

**Not a runnable or validated task.** Copy this directory outside `templates/`,
then implement every TODO before including it in a dataset. The placeholder
environment and verifier deliberately fail instead of producing fake rewards.

1. Implement a native RobotWorld episode adapter and its contract tests.
2. Replace the environment image/build instructions and asset mounts. Keep the
   agent (`main`), simulator and trusted verifier separate.
3. Specify public task goals in `instruction.md`. Do not reveal hidden checker
   state, provider credentials or a privileged reference solution.
4. Fill `task.toml` from an accepted task, with native control-step budget in the
   simulator configuration and independent wall-clock limits. Declare MCP.
5. Collect simulator-owned artifacts and grade them in the separate verifier.
6. Test success, non-success, timeouts and invalid simulations. Only then register
   the package and optionally add a validated `solution/solve.sh`.

这是不能直接运行的造题模板；请先实现原生环境适配与评分，再加入正式题库。
模板不提供假成功，也不能用作任务已验证的证明。

See [English guide](../../docs/HARBOR.md) / [中文指南](../../docs/HARBOR.zh-CN.md).
