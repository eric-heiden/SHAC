import warp as wp
import torch


@wp.kernel
def assign_kernel(
    b: wp.array(dtype=float),
    # outputs
    a: wp.array(dtype=float),
):
    tid = wp.tid()
    a[tid] = b[tid]


def float_assign(a, b):
    wp.launch(
        assign_kernel,
        dim=len(b),
        device=b.device,
        inputs=[b],
        outputs=[a],
    )
    return a


@wp.kernel
def assign_kernel_2d(
    b: wp.array2d(dtype=float),
    # outputs
    a: wp.array(dtype=float),
):
    tid = wp.tid()
    a[2 * tid] = b[tid, 0]
    a[2 * tid + 1] = b[tid, 1]


def float_assign_2d(a, b):
    wp.launch(
        assign_kernel_2d,
        dim=len(b),
        device=b.device,
        inputs=[b],
        outputs=[a],
    )
    return a


@wp.kernel
def assign_act_kernel(
    b: wp.array2d(dtype=float),
    # outputs
    a: wp.array(dtype=float),
):
    tid = wp.tid()
    a[2 * tid] = b[tid, 0]


def float_assign_joint_act(a, b):
    wp.launch(
        assign_act_kernel,
        dim=len(b),
        device=b.device,
        inputs=[b],
        outputs=[a],
    )
    return a


class KernelAutogradFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        integrator,
        model,
        state_in,
        dt,
        substeps,
        *tensors,
    ):
        ctx.tape = wp.Tape()
        ctx.model = model
        ctx.inputs = [wp.from_torch(t) for t in tensors]
        # allocate output

        with ctx.tape:
            float_assign_joint_act(ctx.model.joint_act, ctx.act)
            float_assign_2d(ctx.model.joint_q, ctx.joint_q_start)
            float_assign_2d(ctx.model.joint_qd, ctx.joint_qd_start)
            # updates body position/vel
            for _ in range(substeps):
                state_out = model.state(requires_grad=True)
                state_in = integrator.simulate(
                    model, state_in, state_out, dt / float(substeps)
                )
            ctx.state_out = state_in
            # updates joint_q joint_qd
            ctx.joint_q_end, ctx.joint_qd_end = model.joint_q, model.joint_qd
            wp.sim.eval_ik(ctx.model, ctx.state_out, ctx.joint_q_end, ctx.joint_qd_end)

        ctx.outputs = to_weak_list(
            ctx.state_out.flatten() + [ctx.joint_q_end, ctx.joint_qd_end]
        )
        return tuple([wp.to_torch(x) for x in ctx.outputs])

    @staticmethod
    def backward(ctx, *adj_outputs):  # , adj_joint_q, adj_joint_qd):
        for adj_out, out in zip(adj_outputs, ctx.outputs):
            out.grad = wp.from_torch(adj_out)
        ctx.tape.backward()
        adj_inputs = [ctx.tape.get(x, None) for x in self.inputs]
        return (None, None, None, None, None, *filter_grads(adj_inputs))


class IntegratorSimulate(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        act,
        joint_q_start,
        joint_qd_start,
        model,
        integrator,
        sim_dt,
    ):
        ctx.tape = wp.Tape()
        ctx.model = model
        ctx.act = wp.from_torch(act)
        ctx.joint_q_start = wp.from_torch(joint_q_start)
        ctx.joint_qd_start = wp.from_torch(joint_qd_start)

        ctx.act.requires_grad = True
        ctx.joint_q_start.requires_grad = True
        ctx.joint_qd_start.requires_grad = True

        ctx.model.joint_q.requires_grad = True
        ctx.model.joint_qd.requires_grad = True
        ctx.model.joint_act.requires_grad = True
        ctx.model.body_q.requires_grad = True
        ctx.model.body_qd.requires_grad = True

        # allocate output
        ctx.state_temp = model.state(requires_grad=True)
        ctx.state_out = model.state(requires_grad=True)

        ctx.out_q = wp.zeros(len(model.joint_q), dtype=wp.float32, requires_grad=True)
        ctx.out_qd = wp.zeros(len(model.joint_qd), dtype=wp.float32, requires_grad=True)

        with ctx.tape:        
            float_assign_joint_act(ctx.model.joint_act, ctx.act)
            float_assign_2d(ctx.model.joint_q, ctx.joint_q_start)
            float_assign_2d(ctx.model.joint_qd, ctx.joint_qd_start)
            # applies input q, qd
            wp.sim.eval_fk(ctx.model, ctx.model.joint_q, ctx.model.joint_qd, None, ctx.state_temp)
            # updates body position/vel, applies input control forces
            ctx.state_out = integrator.simulate(ctx.model, ctx.state_temp, ctx.state_out, sim_dt)
            # updates q, qd
            wp.sim.eval_ik(ctx.model, ctx.state_out, ctx.out_q, ctx.out_qd)

        import numpy as np
        ctx.tape.backward(grads={
            ctx.out_q: wp.array(np.ones((len(ctx.out_q)), dtype=np.float32)),
            ctx.out_qd: wp.array(np.ones((len(ctx.out_qd)), dtype=np.float32)),
        })
        ctx.joint_act_grad = wp.to_torch(ctx.tape.gradients[ctx.act]).clone()
        ctx.joint_q_grad = wp.to_torch(ctx.tape.gradients[ctx.model.joint_q]).clone().view(-1, 2)
        ctx.joint_qd_grad = wp.to_torch(ctx.tape.gradients[ctx.joint_qd_start]).clone().view(-1, 2)

        print("joint_act_grad", ctx.joint_act_grad)
        print("joint_q_grad", ctx.joint_q_grad)
        print("joint_qd_grad", ctx.joint_qd_grad)
        ctx.tape.zero()

        return (
            wp.to_torch(ctx.out_q).view(-1, 2),
            wp.to_torch(ctx.out_qd).view(-1, 2),
        )

    @staticmethod
    def backward(ctx, adj_joint_q, adj_joint_qd):

        # map incoming Torch grads to our output variables
        # ctx.out_q.grad = wp.from_torch(adj_joint_q.flatten())
        # ctx.out_qd.grad = wp.from_torch(adj_joint_qd.flatten())

        # ctx.tape.backward(grads={
        #     ctx.out_q: wp.from_torch(adj_joint_q.flatten()),
        #     ctx.out_qd: wp.from_torch(adj_joint_qd.flatten()),
        # })
        # import numpy as np
        # ctx.tape.backward(grads={
        #     ctx.out_q: wp.array(np.ones((len(ctx.out_q)), dtype=np.float32)),
        #     ctx.out_qd: wp.array(np.ones((len(ctx.out_qd)), dtype=np.float32)),
        # })
        # joint_act_grad = wp.to_torch(ctx.tape.gradients[ctx.act]).clone()
        # joint_q_grad = wp.to_torch(ctx.tape.gradients[ctx.joint_q_start]).clone()
        # joint_qd_grad = wp.to_torch(ctx.tape.gradients[ctx.joint_qd_start]).clone()

        # ctx.tape.zero()

        # return adjoint w.r.t. inputs
        return (
            ctx.joint_act_grad,
            ctx.joint_q_grad,
            ctx.joint_qd_grad,
            None,
            None,
            None,
        )
