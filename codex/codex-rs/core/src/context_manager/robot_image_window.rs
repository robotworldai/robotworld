//! Opt-in RobotWorld image elision for the active history, not the rollout archive.
//! The external trusted adapter still validates packet hashes and timestamps.

use codex_history::ResponseItemEnvelope;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ResponseItem;
use std::collections::HashSet;

const OLD_PACKET: &str = "[Earlier packet image removed by visual-memory policy; use the latest CURRENT/HISTORY packet. Original frame remains in the run archive.]";
const OLD_VIEW: &str = "[Earlier view_image removed by visual-memory policy; only the newest four auxiliary images since the current observation packet are retained, within the 50-image cap.]";

fn image_count(item: &ResponseItem) -> usize {
    match item {
        ResponseItem::Message { content, .. } => content
            .iter()
            .filter(|p| matches!(p, ContentItem::InputImage { .. }))
            .count(),
        ResponseItem::FunctionCallOutput { output, .. } => match &output.body {
            FunctionCallOutputBody::ContentItems(parts) => parts
                .iter()
                .filter(|p| matches!(p, FunctionCallOutputContentItem::InputImage { .. }))
                .count(),
            FunctionCallOutputBody::Text(_) => 0,
        },
        _ => 0,
    }
}

fn current_label(text: &str) -> bool {
    let Some(rest) = text.strip_prefix("CURRENT observation ") else {
        return false;
    };
    let Some((round, step)) = rest.split_once("; env step ") else {
        return false;
    };
    round.parse::<u64>().is_ok() && step.parse::<u64>().is_ok()
}

fn is_packet(item: &ResponseItem) -> bool {
    match item {
        ResponseItem::Message { role, content, .. } if role == "user" => content
            .iter()
            .any(|p| matches!(p, ContentItem::InputText { text } if current_label(text))),
        ResponseItem::FunctionCallOutput { output, .. } => match &output.body {
            FunctionCallOutputBody::ContentItems(parts) => parts.iter().any(|p| {
                matches!(p, FunctionCallOutputContentItem::InputText { text } if current_label(text))
            }),
            FunctionCallOutputBody::Text(_) => false,
        },
        _ => false,
    }
}

fn packet_order(item: &ResponseItem) -> Option<(u64, u64)> {
    fn parse(text: &str) -> Option<(u64, u64)> {
        let (round, step) = text
            .strip_prefix("CURRENT observation ")?
            .split_once("; env step ")?;
        Some((round.parse().ok()?, step.parse().ok()?))
    }
    let labels: Vec<_> = match item {
        ResponseItem::Message { role, content, .. } if role == "user" => content
            .iter()
            .filter_map(|p| {
                if let ContentItem::InputText { text } = p {
                    parse(text)
                } else {
                    None
                }
            })
            .collect(),
        ResponseItem::FunctionCallOutput { output, .. } => match &output.body {
            FunctionCallOutputBody::ContentItems(parts) => parts
                .iter()
                .filter_map(|p| {
                    if let FunctionCallOutputContentItem::InputText { text } = p {
                        parse(text)
                    } else {
                        None
                    }
                })
                .collect(),
            _ => Vec::new(),
        },
        _ => Vec::new(),
    };
    if labels.len() == 1 {
        Some(labels[0])
    } else {
        None
    }
}

fn elide(item: &mut ResponseItem, mut count: usize, notice: &str) {
    match item {
        ResponseItem::Message { content, .. } => {
            for part in content {
                if count > 0 && matches!(part, ContentItem::InputImage { .. }) {
                    *part = ContentItem::InputText {
                        text: notice.into(),
                    };
                    count -= 1;
                }
            }
        }
        ResponseItem::FunctionCallOutput { output, .. } => {
            if let FunctionCallOutputBody::ContentItems(parts) = &mut output.body {
                for part in parts {
                    if count > 0 && matches!(part, FunctionCallOutputContentItem::InputImage { .. })
                    {
                        *part = FunctionCallOutputContentItem::InputText {
                            text: notice.into(),
                        };
                        count -= 1;
                    }
                }
            }
        }
        _ => {}
    }
}

pub(super) fn prune(items: &mut [ResponseItemEnvelope]) -> usize {
    let mut view_calls = HashSet::new();
    let mut locations = Vec::new();
    for (index, envelope) in items.iter().enumerate() {
        if let ResponseItem::FunctionCall {
            name,
            namespace,
            call_id,
            ..
        } = &envelope.item
            && name == "view_image"
            && namespace.as_deref().is_none_or(|ns| ns == "functions")
        {
            view_calls.insert(call_id.clone());
        }
        let count = image_count(&envelope.item);
        if count == 0 {
            continue;
        }
        let view = matches!(&envelope.item,
            ResponseItem::FunctionCallOutput { call_id: Some(id), .. } if view_calls.contains(id));
        locations.push((index, count, view));
    }
    let packets: Vec<_> = locations.iter().filter(|(_, _, view)| !view).collect();
    // Concurrent calls may be recorded in call order rather than physical
    // completion order. Retain the newest explicitly numbered observation.
    if packets
        .iter()
        .any(|(i, _, _)| packet_order(&items[*i].item).is_none())
    {
        return 0;
    }
    let Some(&&(latest, packet_count, _)) = packets
        .iter()
        .max_by_key(|(i, _, _)| (packet_order(&items[*i].item), *i))
    else {
        return 0;
    };
    // Never repair a missing/unknown/latest malformed packet by choosing an older one.
    // Leave it intact for the external adapter's fail-closed validation.
    if !is_packet(&items[latest].item) || packet_count > 50 {
        return 0;
    }
    let extras: usize = locations
        .iter()
        .filter(|(i, _, view)| *view && *i > latest)
        .map(|(_, count, _)| count)
        .sum();
    let capacity = 4.min(50 - packet_count);
    if extras > 0 && capacity == 0 {
        return 0;
    }
    let mut extra_remove = extras.saturating_sub(capacity);
    let mut removed = 0;
    for (index, count, view) in locations {
        let remove = if !view && index != latest || view && index < latest {
            count
        } else if view {
            let remove = count.min(extra_remove);
            extra_remove -= remove;
            remove
        } else {
            0
        };
        if remove > 0 {
            elide(
                &mut items[index].item,
                remove,
                if view { OLD_VIEW } else { OLD_PACKET },
            );
            removed += remove;
        }
    }
    removed
}

#[cfg(test)]
#[path = "robot_image_window_tests.rs"]
mod tests;
