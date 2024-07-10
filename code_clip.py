       /opt/anaconda3/envs/llava/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py
	LlamaSdpaAttention


        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            import json; data = json.load(open('/sunrui/torus/llava/LLaVA/hucfg.json')); alpha1 = float(data['alpha1']);alpha2 = float(data['alpha2'])
            alpha = alpha1
            key_states[:,:,1:,:] = key_states[:,:,1:,:]*(1-alpha) + key_states[:,:,:-1,:]*alpha
            alpha = alpha2
            value_states[:,:,1:,:] = value_states[:,:,1:,:]*(1-alpha) + value_states[:,:,:-1,:]*alpha
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)



                if i == 0:
                    noise_momentum = noise_pred
                alpha =  0.9
                noise_momentum = noise_momentum*alpha + noise_pred*(1- alpha)
                # import ipdb; ipdb.set_trace()
                noise_pred = torch.sigmoid(-noise_momentum*noise_pred)*0.1*noise_pred + noise_pred
                # noise_pred = (2*noise_pred - noise_momentum)


