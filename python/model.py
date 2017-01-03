import copy

TIMESTEP=.0001

class Compartment(object):
    
    def __init__(self,forward_neigh,back_neigh,forward_rate_const,back_rate_const,init_val):
        self.forward_neigh=forward_neigh
        self.back_neigh=back_neigh
        self.forward_rate_const=forward_rate_const
        self.back_rate_const=back_rate_const
        self.init_val=init_val
        self.cur_val=init_val
        self.next_add=0.0

    def reset(self):
        self.cur_val=init_val
    
    def time_step(self):
        forward_change=-self.cur_val*self.forward_rate_const*TIMESTEP
        back_change=-self.cur_val*self.back_rate_const*TIMESTEP
        self.cur_val+=(forward_change+back_change)
        self.forward_neigh.add_future(forward_change)
        self.back_neigh.add_future(back_change)
        self.cur_val+=self.next_add
        self.next_add=0.0
    
    def add_future(self,val):
        self.next_add+=val

class Model(object):
    
    """init_vals is a list of tuples where each element in the list is a compartment, and each element in the tuple is a sub compartment
    rate_consts is a list of list of tuples where each element in the list corresponds to the compartment, and each tuple within the tuple corresponds to (forward_flow, back_flow)
    """
    def __init__(self,):
        

if __name__=='__main__':
    model=Model([[.27,.73]],rate_consts=[[[0.0,8.14],[0.0,.057]]])#Corresponds to R1, R2,, no F, no backflow, and unknown forward rate flow for each
    model.predict(5.0)
    
    
    
    
