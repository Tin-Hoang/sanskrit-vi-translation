import pandas as pd
from pathlib import Path
import re


def clean_str(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def get_han_viet_aligned(raw_lines):
    # Join all lines 23-45 (0-indexed: 22-44)
    # Actually let's use the strings from the file content we know
    full_text = " ".join([clean_str(l) for l in raw_lines[22:45]])

    # Define split markers or regex suitable for Heart Sutra Han-Viet
    # 1. Avalokiteshvara...
    # 2. Form is emptiness...
    # 3. Same for feeling...
    # 4. Sariputra, all dharmas...
    # 5. Therefore in emptiness...
    # 6. No eye, ear...
    # 7. No form, sound...
    # 8. No eye-element...
    # 9. No ignorance...
    # 10. No suffering...
    # 11. No wisdom...
    # 12. Therefore... (Bodhisattva relies on Prajna)
    # 13. No hindrance...
    # 14. All Buddhas...
    # 15. Therefore know... (Mantra intro)
    # 16. Relieve all suffering
    # 17. The mantra says:
    # 18. Gate gate...

    # We will use simple keyword splitting or list construction
    segments = [
        "Quán Tự Tại Bồ Tát hành thâm Bát nhã Ba la mật đa thời, chiếu kiến ngũ uẩn giai không, độ nhất thiết khổ ách.",
        "Xá Lợi Tử, sắc bất dị không, không bất dị sắc, sắc tức thị không, không tức thị sắc,",
        "thọ tưởng hành thức diệc phục như thị.",
        "Xá Lợi Tử, thị chư pháp không tướng, bất sanh bất diệt, bất cấu bất tịnh, bất tăng bất giảm.",
        "Thị cố không trung vô sắc, vô thọ tưởng hành thức.",
        "Vô nhãn nhĩ tỷ thiệt thân ý,",
        "vô sắc thanh hương vị xúc pháp,",
        "vô nhãn giới nãi chí vô ý thức giới.",
        "Vô vô minh, diệc vô vô minh tận, nãi chí vô lão tử, diệc vô lão tử tận.",
        "Vô khổ, tập, diệt, đạo.",
        "Vô trí diệc vô đắc, dĩ vô sở đắc cố.",
        "Bồ đề tát đõa y Bát nhã Ba la mật đa cố, tâm vô quái ngại,",
        "vô quái ngại cố, vô hữu khủng bố, viễn ly điên đảo mộng tưởng, cứu cánh Niết bàn.",
        "Tam thế chư Phật, y Bát nhã Ba la mật đa cố, đắc A nậu đa la Tam miệu Tam bồ đề.",
        "Cố tri Bát nhã Ba la mật đa, thị đại thần chú, thị đại minh chú, thị vô thượng chú, thị vô đẳng đẳng chú,",
        "năng trừ nhất thiết khổ, chân thật bất hư.",
        "Cố thuyết Bát nhã Ba la mật đa chú, tức thuyết chú viết:",
        "Yết đế yết đế, ba la yết đế, ba la tăng yết đế, bồ đề tát bà ha.",
    ]
    return segments


def get_modern_aligned(raw_lines):
    # Modern version lines 46-71 (0-idx: 45-70)
    # We reconstruct based on semantic breaks matching the Sanskrit 18
    # This is "approximate" but better than random

    segments = [
        "Ngài Bồ Tát Quán Tự Tại khi thực hành thâm sâu về trí tuệ Bát Nhã Ba la mật, thì soi thấy năm uẩn đều là không, do đó vượt qua mọi khổ đau ách nạn.",
        "Nầy Xá Lợi Tử, sắc chẳng khác gì không, không chẳng khác gì sắc, sắc chính là không, không chính là sắc,",
        "thọ tưởng hành thức cũng đều như thế.",
        "Nầy Xá Lợi Tử, tướng không của các pháp ấy chẳng sinh chẳng diệt, chẳng nhơ chẳng sạch, chẳng thêm chẳng bớt.",
        "Cho nên trong cái không đó, nó không có sắc, không thọ tưởng hành thức.",
        "Không có mắt, tai, mũi, lưỡi, thân ý.",
        "Không có sắc, thanh, hương vị, xúc pháp.",
        "Không có nhãn giới cho đến không có ý thức giới.",
        "Không có vô minh, mà cũng không có hết vô minh. Không có già chết, mà cũng không có hết già chết.",
        "Không có khổ, tập, diệt, đạo.",
        "Không có trí cũng không có đắc, vì không có sở đắc.",
        "Khi vị Bồ Tát nương tựa vào trí tuệ Bát Nhã nầy thì tâm không còn chướng ngại,",
        "vì tâm không chướng ngại nên không còn sợ hãi, xa lìa được cái điên đảo mộng tưởng, đạt đến cứu cánh Niết Bàn.",
        "Các vị Phật ba đời vì nương theo trí tuệ Bát Nhã nầy mà đắc quả vô thượng, chánh đẳng chánh giác.",
        "Cho nên phải biết rằng Bát nhã Ba la mật đa là đại thần chú, là đại minh chú, là chú vô thượng, là chú cao cấp nhất,",
        "luôn trừ các khổ não, chân thật không hư dối.",
        "Cho nên khi nói đến Bát nhã Ba la mật đa, tức là phải nói câu chú:",
        "Yết đế yết đế, ba la yết đế, ba la tăng yết đế, bồ đề tát bà ha.",
    ]
    return segments


def main():
    raw_path = Path("sanskrit-vi-translation/data/crawled_raw.csv")
    if not raw_path.exists():
        print("No raw data")
        return

    df_raw = pd.read_csv(raw_path)
    raw_lines = df_raw["raw_text"].tolist()

    sanskrit_18 = [
        "ārya-avalokiteśvaro bodhisattvo gambhīrāṃ prajñāpāramitā caryāṃ caramāṇo vyavalokayati sma: panca-skandhās tāṃś ca svābhava śūnyān paśyati sma.",
        "iha śāriputra: rūpaṃ śūnyatā śūnyataiva rūpaṃ; rūpān na pṛthak śūnyatā śunyatāyā na pṛthag rūpaṃ; yad rūpaṃ sā śūnyatā; ya śūnyatā tad rūpaṃ.",
        "evam eva vedanā saṃjñā saṃskāra vijñānaṃ.",
        "iha śāriputra: sarva-dharmāḥ śūnyatā-lakṣaṇā, anutpannā aniruddhā, amalā avimalā, anūnā aparipūrṇāḥ.",
        "tasmāc chāriputra śūnyatayāṃ na rūpaṃ na vedanā na saṃjñā na saṃskārāḥ na vijñānam.",
        "na cakṣuḥ-śrotra-ghrāna-jihvā-kāya-manāṃsi.",
        "na rūpa-śabda-gandha-rasa-spraṣṭavya-dharmaah.",
        "na cakṣūr-dhātur yāvan na manovijñāna-dhātuḥ.",
        "na-avidyā na-avidyā-kṣayo yāvan na jarā-maraṇam na jarā-maraṇa-kṣayo.",
        "na duhkha-samudaya-nirodha-margā.",
        "na jñānam, na prāptir na-aprāptiḥ.",
        "tasmāc chāriputra aprāptitvād bodhisattvasya prajñāpāramitām āśritya viharatyacittāvaraṇaḥ.",
        "cittāvaraṇa-nāstitvād atrastro viparyāsa-atikrānto niṣṭhā-nirvāṇa-prāptaḥ.",
        "tryadhva-vyavasthitāḥ sarva-buddhāḥ prajñāpāramitām āśrityā-anuttarāṃ samyaksambodhim abhisambuddhāḥ.",
        "tasmāj jñātavyam: prajñāpāramitā mahā-mantro mahā-vidyā mantro 'nuttara-mantro samasama-mantraḥ.",
        "sarva duḥkha praśamanaḥ, satyam amithyatāt.",
        "prajñāpāramitāyām ukto mantraḥ tadyathā:",
        "gate gate pāragate pārasaṃgate bodhi svāhā.",
    ]

    han_viet = get_han_viet_aligned(raw_lines)
    modern = get_modern_aligned(raw_lines)

    # Pad others with empty for now to ensure 18 rows
    poetic = [""] * 18
    scholarly = [""] * 18

    aligned_data = []
    for i in range(18):
        aligned_data.append(
            {
                "id": i + 1,
                "sanskrit_text": sanskrit_18[i],
                "ref_han_viet": han_viet[i] if i < len(han_viet) else "",
                "ref_viet_modern": modern[i] if i < len(modern) else "",
                "source": "Heart Sutra (Budsas - Manual Align)",
            }
        )

    out_path = Path("sanskrit-vi-translation/data/sanskrit_vi_heart_sutra.csv")
    pd.DataFrame(aligned_data).to_csv(out_path, index=False)
    print(f"Saved manually aligned dataset to {out_path}")
    print("Sanity Check Link 18:")
    print(aligned_data[-1])


if __name__ == "__main__":
    main()
