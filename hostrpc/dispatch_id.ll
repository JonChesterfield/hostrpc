target triple = "amdgcn-amd-amdhsa"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-ni:7"

declare i64 @llvm.amdgcn.dispatch.id() #0

attributes #0 = { nounwind readnone speculatable }

define i64 @my_dispatch_id() {
  %val = call i64 @llvm.amdgcn.dispatch.id()
  ret i64 %val
}
