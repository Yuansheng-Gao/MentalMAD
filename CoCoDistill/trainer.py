import torch
from sklearn.metrics import classification_report

def train(phase,
          epoch, 
          model, 
          dataloader, 
          optimizer, 
          accelerator, 
          gradient_accumulation_steps, 
          log_interval = 10,
          logger = None):
    model.train()
    accumulation_steps = 0
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100,reduction='mean')

    for i, batch in enumerate(dataloader):
        with accelerator.autocast():

            outputs_rationale = model(
                input_ids=batch["rationale_input_ids"].to(accelerator.device),
                attention_mask=batch["rationale_attention_mask"].to(accelerator.device),
                labels=batch["rationale_labels"].to(accelerator.device)
            )

            logits_rationale = outputs_rationale.logits[:, :-1, :].contiguous()
            labels_rationale = batch["rationale_labels"][:, 1:].contiguous().to(accelerator.device)

            loss_rationale = loss_fct(
               logits_rationale.view(-1, logits_rationale.size(-1)),
               labels_rationale.view(-1)
            )

            valid_token_mask = labels_rationale != -100

            token_order = valid_token_mask.float().cumsum(dim=1)

            binary_mask = (token_order >= 1) & (token_order <= 1)

            loss_binary = loss_fct(
                logits_rationale[binary_mask].view(-1, logits_rationale.size(-1)),
                labels_rationale[binary_mask].view(-1)
            )

            del logits_rationale, outputs_rationale, labels_rationale

            outputs_binary = model(
                input_ids=batch["binary_input_ids"].to(accelerator.device),
                attention_mask=batch["binary_attention_mask"].to(accelerator.device),
                labels=batch["binary_labels"].to(accelerator.device)
            )
            
            logits_binary = outputs_binary.logits[:, :-1, :].contiguous()
            labels_binary = batch["binary_labels"][:, 1:].contiguous().to(accelerator.device)

            loss_binary = loss_fct(
                logits_binary.view(-1, logits_binary.size(-1)),
                labels_binary.view(-1)
                )
            
            del logits_binary, outputs_binary, labels_binary

            if phase == 1:
                outputs_feedback = model(
                    input_ids=batch["feedback_input_ids"].to(accelerator.device),
                    attention_mask=batch["feedback_attention_mask"].to(accelerator.device),
                    labels=batch["feedback_labels"].to(accelerator.device)
                )
                
                logits_feedback = outputs_feedback.logits[:, :-1, :].contiguous()
                labels_feedback = batch["feedback_labels"][:, 1:].contiguous().to(accelerator.device)

                loss_feedback = loss_fct(
                    logits_feedback.view(-1, logits_feedback.size(-1)),
                    labels_feedback.view(-1)
                    )
                
                del logits_feedback, outputs_feedback, labels_feedback

                loss = loss_binary + loss_rationale + loss_feedback
            if phase == 2:
                loss = loss_binary + loss_rationale
            if phase == 3:
                loss = loss_binary

        if i % log_interval == 0:
            msg = (f"[Training] Phase: {str(phase)}, Epoch: {epoch}, step: {i}, loss: {loss.item():.4f}")
            if logger:
                logger.info(msg)
            else:
                print(msg)

        accelerator.backward(loss)
        accumulation_steps += 1

        if accumulation_steps % gradient_accumulation_steps == 0:
            with accelerator.accumulate(model):
                accelerator.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()
            accumulation_steps = 0


def extract_label_from_response(response):
    response = response.lower()
    if "yes" in response:
        return 1
    else:
        return 0
    
def validate(
    tokenizer, 
    model, 
    dataloader,
    accelerator,
    max_new_tokens=1,
    temperature=0.1,
    top_p=0.95,
    top_k=50,
    do_sample=False
):
    """
    Evaluate model with datasets formatted like ValidDataset (input_ids/attention_mask/labels).
    Only the prompt segment (where labels == -100 and attention_mask == 1) is used as conditioning.
    """
    model.eval()
    predictions, actuals = [], []
    device = accelerator.device

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # The prompt length of each sample.
            prompt_lens = ((labels == -100) & (attention_mask == 1)).sum(dim=1).tolist()

            batch_preds = []
            for ids, p_len in zip(input_ids, prompt_lens):
                # Only keep the prompt section.
                prompt_ids = ids[:p_len].unsqueeze(0)
                prompt_mask = torch.ones_like(prompt_ids)

                gen = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=tokenizer.eos_token_id,
                )
                # Remove the input section and only take the new additions.
                new_tokens = gen[0, prompt_ids.size(1):]
                batch_preds.append(new_tokens.tolist())

            predictions.extend(tokenizer.batch_decode(batch_preds, skip_special_tokens=True))

            y_fixed = labels.clone()
            y_fixed[y_fixed == -100] = pad_id
            actuals.extend(tokenizer.batch_decode(y_fixed, skip_special_tokens=True))
            

    y_pred = [extract_label_from_response(r) for r in predictions]
    y_true = [extract_label_from_response(r) for r in actuals]
    report = classification_report(y_true, y_pred, output_dict=True)
    return report
