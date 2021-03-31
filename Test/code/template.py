def set_template(args):
        
    #baseline
    if args.template.find('LPNet') >= 0:
        args.model = 'LPNet'
        args.n_feats = 32
        args.kernel_size = 3