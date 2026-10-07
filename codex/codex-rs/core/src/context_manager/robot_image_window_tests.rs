use super::*;
use pretty_assertions::assert_eq;
use serde_json::json;

fn item(value: serde_json::Value) -> ResponseItemEnvelope {
    ResponseItemEnvelope::new(serde_json::from_value(value).unwrap())
}

fn packet(round: usize) -> ResponseItemEnvelope {
    item(
        json!({"type":"function_call_output", "call_id":format!("step-{round}"), "output":[
            {"type":"input_text","text":format!("CURRENT observation {round}; env step {round}")},
            {"type":"input_image","image_url":"data:image/png;base64,YQ=="},
            {"type":"input_text","text":"preserved receipt"}
        ]}),
    )
}

#[test]
fn long_history_preserves_archive_and_receipts() {
    let mut history = Vec::new();
    let mut archive = Vec::new();
    for round in 0..1000 {
        let incoming = packet(round);
        archive.push(incoming.clone());
        history.push(incoming.clone());
        prune(&mut history);
        assert_eq!(history.last().unwrap(), &incoming);
        assert_eq!(
            history.iter().map(|p| image_count(&p.item)).sum::<usize>(),
            1
        );
    }
    assert_eq!(
        archive.iter().map(|p| image_count(&p.item)).sum::<usize>(),
        1000
    );
    let mut expected = archive.clone();
    for envelope in &mut expected[..999] {
        elide(&mut envelope.item, 1, OLD_PACKET);
    }
    assert_eq!(history, expected);
    assert_eq!(prune(&mut history), 0);
}

#[test]
fn auxiliary_images_keep_latest_four_and_expire_at_next_packet() {
    let mut history = vec![packet(0)];
    for i in 0..6 {
        history.push(item(
            json!({"type":"function_call","name":"view_image", "namespace":"functions",
            "call_id":format!("crop-{i}"),"arguments":"{}"}),
        ));
        history.push(item(
            json!({"type":"function_call_output","call_id":format!("crop-{i}"),
            "output":[{"type":"input_image","image_url":"data:image/png;base64,Yg=="}]}),
        ));
        prune(&mut history);
    }
    assert_eq!(
        history.iter().map(|p| image_count(&p.item)).sum::<usize>(),
        5
    );
    let calls: Vec<_> = history
        .iter()
        .filter(|p| matches!(p.item, ResponseItem::FunctionCall { .. }))
        .cloned()
        .collect();
    history.push(packet(1));
    prune(&mut history);
    assert_eq!(
        history.iter().map(|p| image_count(&p.item)).sum::<usize>(),
        1
    );
    assert_eq!(
        history
            .iter()
            .filter(|p| matches!(p.item, ResponseItem::FunctionCall { .. }))
            .cloned()
            .collect::<Vec<_>>(),
        calls
    );
}

#[test]
fn unknown_latest_image_is_left_for_external_validator() {
    let mut history = vec![
        packet(0),
        item(json!({"type":"message","role":"user", "content":[
        {"type":"input_image","image_url":"data:image/png;base64,Yg=="}]})),
    ];
    let original = history.clone();
    assert_eq!(prune(&mut history), 0);
    assert_eq!(history, original);
}

#[test]
fn out_of_order_receipts_keep_newest_observation() {
    let newest = packet(27);
    let mut history = vec![packet(15), newest.clone()];
    prune(&mut history);
    history.push(packet(23));
    prune(&mut history);
    assert_eq!(history[1], newest);
    assert_eq!(image_count(&history[2].item), 0);
    assert_eq!(
        history.iter().map(|p| image_count(&p.item)).sum::<usize>(),
        1
    );
}
