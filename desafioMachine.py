import numpy as np

# Exemplo de matriz de confusão
# Substitua TN, FP, FN, TP pelos valores reais
con_mat = np.array([[50, 10], [5, 35]])  

class Metrics:
    def __init__(self, con_mat):
        self.VP = con_mat[1, 1]
        self.VN = con_mat[0, 0]
        self.FP = con_mat[0, 1]
        self.FN = con_mat[1, 0]
    
    def acuracia(self):
        return (self.VP + self.VN) / (self.VP + self.VN + self.FP + self.FN)
    
    def sensibilidade(self):
        return self.VP / (self.VP + self.FN)
    
    def especificidade(self):
        return self.VN / (self.VN + self.FP)
    
    def precisao(self):
        return self.VP / (self.VP + self.FP)
    
    def f_score(self):
        prec = self.precisao()
        rec = self.sensibilidade()
        return 2 * (prec * rec) / (prec + rec)

# Instanciar a classe Metrics para o uso e calcular
metricas = Metrics(con_mat)

acc = metricas.acuracia()
rec = metricas.sensibilidade()  # Corrigido
spec = metricas.especificidade()
prec = metricas.precisao()      # Corrigido
fk = metricas.f_score()

print(f"Acurácia: {acc:.2f}")
print(f"Sensibilidade: {rec:.2f}")
print(f"Especificidade: {spec:.2f}")
print(f"Precisão: {prec:.2f}")
print(f"F-Score: {fk:.2f}")
