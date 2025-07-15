# 查看集群信息
kubectl cluster-info

# 查看节点状态
kubectl get nodes

# 查看所有命名空间下的Pod
kubectl get pods --all-namespaces

# 从docker目录部署服务（示例）
kubectl apply -f ./docker/server/deployment.yaml

# 查看部署状态
kubectl get deployments

# 滚动更新镜像
kubectl set image deployment/my-app my-container=my-image:1.1

# 查看Pod日志
kubectl logs <pod-name> --tail=100

# 进入容器终端
kubectl exec -it <pod-name> -- /bin/bash

# 查看服务详情
kubectl describe service my-service

# 创建ConfigMap
kubectl create configmap app-config --from-literal=KEY=VALUE

# 创建Secret（从文件）
kubectl create secret generic db-secret --from-file=username.txt --from-file=password.txt

# 查看事件日志
kubectl get events --sort-by=.metadata.creationTimestamp

# 检查资源使用情况
kubectl top node
kubectl top pod