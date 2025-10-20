import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import negative_sampling


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE编码器
    用于学习艺人节点的嵌入表示
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.2):
        """
        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出嵌入维度
            num_layers: 层数
            dropout: Dropout率
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # 第一层
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # 输出层
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        
    def forward(self, x, edge_index):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            节点嵌入 [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        return x


class GATEncoder(nn.Module):
    """
    GAT (Graph Attention Network) 编码器
    使用注意力机制学习节点嵌入
    """
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 heads: int = 4, dropout: float = 0.2):
        """
        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度
            out_channels: 输出嵌入维度
            heads: 注意力头数
            dropout: Dropout率
        """
        super().__init__()
        
        self.dropout = dropout
        
        # 第一层GAT (多头注意力)
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        
        # 第二层GAT (输出单头)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, 
                            concat=False, dropout=dropout)
        
    def forward(self, x, edge_index):
        """前向传播"""
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        
        return x


class LinkPredictor(nn.Module):
    """
    链接预测器
    预测两个艺人合作的成功度
    """
    
    def __init__(self, in_channels: int, hidden_channels: int = 128):
        """
        Args:
            in_channels: 输入嵌入维度
            hidden_channels: MLP隐藏层维度
        """
        super().__init__()
        
        # MLP预测器
        self.lin1 = nn.Linear(in_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.lin3 = nn.Linear(hidden_channels // 2, 1)
        
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels // 2)
        
    def forward(self, z_src, z_dst):
        """
        预测边的得分
        
        Args:
            z_src: 源节点嵌入 [num_edges, embedding_dim]
            z_dst: 目标节点嵌入 [num_edges, embedding_dim]
            
        Returns:
            edge_scores: 边的得分 [num_edges]
        """
        # 拼接源和目标节点嵌入
        z = torch.cat([z_src, z_dst], dim=-1)
        
        z = self.lin1(z)
        z = self.bn1(z)
        z = F.relu(z)
        z = F.dropout(z, p=0.2, training=self.training)
        
        z = self.lin2(z)
        z = self.bn2(z)
        z = F.relu(z)
        z = F.dropout(z, p=0.2, training=self.training)
        
        z = self.lin3(z)
        
        return z.squeeze(-1)


class CollaborationGNN(nn.Module):
    """
    完整的艺人合作预测GNN模型
    结合编码器和链接预测器
    """
    
    def __init__(self, in_channels: int, hidden_channels: int = 128, 
                 out_channels: int = 64, encoder_type: str = 'sage',
                 num_layers: int = 2, dropout: float = 0.2):
        """
        Args:
            in_channels: 节点特征维度
            hidden_channels: 隐藏层维度
            out_channels: 节点嵌入维度
            encoder_type: 编码器类型 ('sage' 或 'gat')
            num_layers: 编码器层数
            dropout: Dropout率
        """
        super().__init__()
        
        self.encoder_type = encoder_type
        
        # 选择编码器
        if encoder_type == 'sage':
            self.encoder = GraphSAGEEncoder(
                in_channels, hidden_channels, out_channels, 
                num_layers, dropout
            )
        elif encoder_type == 'gat':
            self.encoder = GATEncoder(
                in_channels, hidden_channels, out_channels, 
                heads=4, dropout=dropout
            )
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        # 链接预测器
        self.predictor = LinkPredictor(out_channels, hidden_channels)
        
    def encode(self, x, edge_index):
        """
        编码节点特征为嵌入
        
        Args:
            x: 节点特征
            edge_index: 边索引
            
        Returns:
            节点嵌入
        """
        return self.encoder(x, edge_index)
    
    def decode(self, z, edge_index):
        """
        解码：预测边的得分
        
        Args:
            z: 节点嵌入
            edge_index: 要预测的边索引 [2, num_edges]
            
        Returns:
            边得分
        """
        src_embeddings = z[edge_index[0]]
        dst_embeddings = z[edge_index[1]]
        
        return self.predictor(src_embeddings, dst_embeddings)
    
    def forward(self, x, edge_index, pred_edge_index):
        """
        完整前向传播
        
        Args:
            x: 节点特征
            edge_index: 训练时使用的边（构建消息传递图）
            pred_edge_index: 要预测的边
            
        Returns:
            预测得分
        """
        z = self.encode(x, edge_index)
        return self.decode(z, pred_edge_index)


class CollaborationGNNWithFeatures(nn.Module):
    """
    增强版GNN模型
    结合节点嵌入和边特征进行预测
    """
    
    def __init__(self, node_in_channels: int, edge_in_channels: int,
                 hidden_channels: int = 128, out_channels: int = 64,
                 encoder_type: str = 'sage', dropout: float = 0.2):
        """
        Args:
            node_in_channels: 节点特征维度
            edge_in_channels: 边特征维度
            hidden_channels: 隐藏层维度
            out_channels: 节点嵌入维度
            encoder_type: 编码器类型
            dropout: Dropout率
        """
        super().__init__()
        
        # 节点编码器
        if encoder_type == 'sage':
            self.encoder = GraphSAGEEncoder(
                node_in_channels, hidden_channels, out_channels, 
                num_layers=2, dropout=dropout
            )
        else:
            self.encoder = GATEncoder(
                node_in_channels, hidden_channels, out_channels, 
                heads=4, dropout=dropout
            )
        
        # 边特征处理MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4)
        )
        
        # 最终预测器（结合节点嵌入和边特征）
        predictor_in_dim = out_channels * 2 + hidden_channels // 4
        self.predictor = nn.Sequential(
            nn.Linear(predictor_in_dim, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
        
    def forward(self, x, edge_index, pred_edge_index, edge_features=None):
        """
        Args:
            x: 节点特征
            edge_index: 消息传递边
            pred_edge_index: 预测边
            edge_features: 边特征（可选）
            
        Returns:
            预测得分
        """
        # 编码节点
        z = self.encoder(x, edge_index)
        
        # 获取预测边的节点嵌入
        src_embeddings = z[pred_edge_index[0]]
        dst_embeddings = z[pred_edge_index[1]]
        
        # 拼接节点嵌入
        combined = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        
        # 如果有边特征，加入预测
        if edge_features is not None:
            edge_embed = self.edge_mlp(edge_features)
            combined = torch.cat([combined, edge_embed], dim=-1)
        
        # 预测
        return self.predictor(combined).squeeze(-1)


def compute_link_loss(pos_pred, neg_pred, loss_type='bce'):
    """
    计算链接预测损失
    
    Args:
        pos_pred: 正样本预测得分
        neg_pred: 负样本预测得分
        loss_type: 损失类型 ('bce' 或 'margin')
        
    Returns:
        损失值
    """
    if loss_type == 'bce':
        # Binary Cross Entropy损失
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_pred, torch.ones_like(pos_pred)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_pred, torch.zeros_like(neg_pred)
        )
        return pos_loss + neg_loss
    
    elif loss_type == 'margin':
        # Margin Ranking损失
        return F.margin_ranking_loss(
            pos_pred, neg_pred, torch.ones(pos_pred.size(0)).to(pos_pred.device),
            margin=1.0
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_regression_loss(pred, target):
    """
    计算回归损失（用于预测具体流行度）
    
    Args:
        pred: 预测值
        target: 真实值
        
    Returns:
        损失值
    """
    # 使用MSE + MAE的组合
    mse_loss = F.mse_loss(pred, target)
    mae_loss = F.l1_loss(pred, target)
    
    return mse_loss + 0.1 * mae_loss


if __name__ == "__main__":
    # 测试模型
    print("Testing CollaborationGNN model...")
    
    # 模拟数据
    num_nodes = 100
    num_edges = 500
    in_channels = 13
    
    x = torch.randn(num_nodes, in_channels)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    pred_edge_index = torch.randint(0, num_nodes, (2, 50))
    
    # 测试GraphSAGE模型
    model_sage = CollaborationGNN(
        in_channels=in_channels,
        hidden_channels=128,
        out_channels=64,
        encoder_type='sage'
    )
    
    output_sage = model_sage(x, edge_index, pred_edge_index)
    print(f"GraphSAGE output shape: {output_sage.shape}")
    
    # 测试GAT模型
    model_gat = CollaborationGNN(
        in_channels=in_channels,
        hidden_channels=128,
        out_channels=64,
        encoder_type='gat'
    )
    
    output_gat = model_gat(x, edge_index, pred_edge_index)
    print(f"GAT output shape: {output_gat.shape}")
    
    # 测试增强模型
    edge_features = torch.randn(50, 4)  # 4个边特征
    model_enhanced = CollaborationGNNWithFeatures(
        node_in_channels=in_channels,
        edge_in_channels=4,
        hidden_channels=128,
        out_channels=64
    )
    
    output_enhanced = model_enhanced(x, edge_index, pred_edge_index, edge_features)
    print(f"Enhanced model output shape: {output_enhanced.shape}")
    
    print("\nAll models tested successfully!")