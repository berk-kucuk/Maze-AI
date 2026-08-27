"""Interface translations.

Maze AI ships with a Turkish interface for Maze Linux's home audience and an
English one for everyone else. The catalogue is keyed by the English source
string, so an untranslated string simply falls through unchanged instead of
showing a placeholder key — a missing entry degrades to English, never to a
broken UI.

Usage::

    from ..i18n import tr
    button.setText(tr("Send"))

Strings with values use ``str.format`` placeholders and are formatted after
translation: ``tr("Running {tool}…").format(tool=name)``.
"""

from __future__ import annotations

import os

#: UI languages offered in Settings. "auto" follows the desktop locale.
UI_LANGUAGES: list[tuple[str, str]] = [
    ("auto", "Auto  ·  match my system"),
    ("en", "English"),
    ("tr", "Türkçe"),
]

TURKISH: dict[str, str] = {
    # ── title bar ─────────────────────────────────────────────────────────
    "Toggle chat history": "Sohbet geçmişini aç/kapat",
    "Reminders": "Hatırlatıcılar",
    "Settings": "Ayarlar",
    "Minimize to tray": "Sistem tepsisine küçült",
    "Hide to tray": "Tepsiye gizle",
    # ── sidebar ───────────────────────────────────────────────────────────
    "New chat": "Yeni sohbet",
    "Search chats…": "Sohbetlerde ara…",
    "HISTORY": "GEÇMİŞ",
    "No saved chats yet.": "Henüz kayıtlı sohbet yok.",
    "No matching chats.": "Eşleşen sohbet yok.",
    "Export chat to Markdown": "Sohbeti Markdown olarak dışa aktar",
    "Delete chat": "Sohbeti sil",
    "Rename chat": "Sohbeti yeniden adlandır",
    "New title:": "Yeni başlık:",
    "Export chat": "Sohbeti dışa aktar",
    "Chat exported to {path}": "Sohbet şuraya aktarıldı: {path}",
    "Export failed: {error}": "Dışa aktarma başarısız: {error}",
    # ── composer ──────────────────────────────────────────────────────────
    "Ask Maze AI to do something…  (Enter to send, Shift+Enter for newline)":
        "Maze AI'dan bir şey yapmasını isteyin…  "
        "(Enter gönderir, Shift+Enter satır atlar)",
    "Click to clear attachments": "Ekleri temizlemek için tıklayın",
    "Attach an image (vision models)": "Görsel ekle (görme destekli modeller)",
    "Attach an image": "Görsel ekle",
    "The current model doesn't support images": "Bu model görselleri desteklemiyor",
    "Attach image(s)": "Görsel ekle",
    "Images (*.png *.jpg *.jpeg *.gif *.webp *.bmp)":
        "Görseller (*.png *.jpg *.jpeg *.gif *.webp *.bmp)",
    "Stop": "Durdur",
    "Send": "Gönder",
    # ── transcript ────────────────────────────────────────────────────────
    "⧉ Copy": "⧉ Kopyala",
    "✓ Copied": "✓ Kopyalandı",
    "code": "kod",
    "Thinking…": "Düşünüyor…",
    "Running {tool}…": "{tool} çalıştırılıyor…",
    "Working…": "Çalışıyor…",
    "Stopping…": "Durduruluyor…",
    "Error:": "Hata:",
    "↻ Regenerate": "↻ Yeniden üret",
    "Re-run the last message": "Son mesajı yeniden çalıştır",
    "Action denied by user.": "İşlem kullanıcı tarafından reddedildi.",
    "{count} image(s) attached": "{count} görsel eklendi",
    "**Maze AI** is ready. I can run commands, manage files, launch apps, fetch "
    "web pages, send notifications and set reminders on Maze Linux.\n\nTry: "
    "*“show my disk usage”*, *“open firefox”*, *“create a python venv in ~/dev”* "
    "or *“remind me to take a break in 30 minutes”*.":
        "**Maze AI** hazır. Maze Linux üzerinde komut çalıştırabilir, dosyaları "
        "yönetebilir, uygulama açabilir, web sayfası getirebilir, bildirim "
        "gönderebilir ve hatırlatıcı kurabilirim.\n\nDeneyin: *“disk kullanımımı "
        "göster”*, *“firefox'u aç”*, *“~/dev içinde python venv oluştur”* ya da "
        "*“30 dakika sonra mola vermemi hatırlat”*.",
    # ── file manager ──────────────────────────────────────────────────────
    "Maze AI": "Maze AI",
    "Ask about this": "Bunu sor",
    "Explain this": "Bunu açıkla",
    "Ask Maze AI": "Maze AI'a sor",
    "Ask Maze AI about this": "Bunu Maze AI'a sor",
    "Hand the selected files to Maze AI": "Seçili dosyaları Maze AI'a ver",
    "No supported file manager found.": "Desteklenen bir dosya yöneticisi bulunamadı.",
    "File manager": "Dosya yöneticisi",
    "Add “Ask Maze AI” to the right-click menu of your file manager.":
        "Dosya yöneticinizin sağ tık menüsüne “Maze AI'a sor” ekler.",
    "Add to file manager": "Dosya yöneticisine ekle",
    "Remove from file manager": "Dosya yöneticisinden kaldır",
    "Added to: {managers}": "Eklendi: {managers}",
    "Removed from: {managers}": "Kaldırıldı: {managers}",
    "Nothing to remove.": "Kaldırılacak bir şey yok.",
    "Could not add the menu: {error}": "Menü eklenemedi: {error}",
    "Restart your file manager to see the new menu.":
        "Yeni menüyü görmek için dosya yöneticinizi yeniden başlatın.",
    "Read the text in this image and explain it: {path}":
        "Bu görseldeki yazıyı oku ve açıkla: {path}",
    "What is in this folder, and what stands out? {path}":
        "Bu klasörde ne var, dikkat çeken ne? {path}",
    "Read this file and summarise it: {path}":
        "Bu dosyayı oku ve özetle: {path}",
    "Tell me what this file is: {path}": "Bu dosya nedir, söyle: {path}",
    "Here are some files. Tell me what they are: {path}":
        "Şu dosyalar var. Ne olduklarını söyle: {path}",
    "Quick Ask": "Hızlı sor",
    "Ask about the clipboard": "Panodakini sor",
    "Ask about a screen area": "Ekrandan bir alanı sor",
    "Shortcuts & terminal": "Kısayollar ve terminal",
    "Bind these to a key in your desktop's shortcut settings, and the "
    "assistant is one keystroke away from anywhere.":
        "Bunları masaüstünüzün kısayol ayarlarında bir tuşa bağlayın; asistan "
        "her yerden tek tuş uzağınızda olsun.",
    "Quick Ask — a floating question bar": "Hızlı sor — yüzen soru çubuğu",
    "Ask about whatever you just copied": "Az önce kopyaladığınız şeyi sorun",
    "Drag a box on screen and ask about it":
        "Ekranda bir alan seçip onu sorun",
    "SHELL INTEGRATION": "KABUK ENTEGRASYONU",
    "Adds `mz` (ask), `mzask` (answer in the terminal) and `mzfix` — "
    "which puts the corrected version of your last failed command "
    "straight onto your prompt.":
        "`mz` (sor), `mzask` (terminalde yanıt) ve `mzfix` ekler — sonuncusu "
        "başarısız olan son komutunuzun düzeltilmiş halini doğrudan komut "
        "satırınıza yazar.",
    # ── quick ask ─────────────────────────────────────────────────────────
    "Ask anything…": "Ne isterseniz sorun…",
    "Asking…": "Soruluyor…",
    "Enter to ask  ·  Esc to close": "Enter ile sor  ·  Esc ile kapat",
    "Ask": "Sor",
    "Continue in chat →": "Sohbette devam et →",
    "The clipboard is empty.": "Pano boş.",
    "What does this show?": "Bu görselde ne var?",
    "Read the text in this image and explain it.":
        "Bu görseldeki yazıyı oku ve açıkla.",
    "Drag a box on the screen…": "Ekranda bir alan seçin…",
    "Taking a screenshot…": "Ekran görüntüsü alınıyor…",
    "Explain": "Açıkla",
    "Fix": "Düzelt",
    "Translate": "Çevir",
    "Summarise": "Özetle",

    # ── local model status ────────────────────────────────────────────────
    "mode": "mod",
    "Ollama is not running": "Ollama çalışmıyor",
    "warming up the model…": "model ısıtılıyor…",
    "native tools": "yerel araç çağrısı",
    "Start Ollama": "Ollama'yı başlat",
    "Run `systemctl --user start ollama` and check again":
        "`systemctl --user start ollama` çalıştırıp tekrar dener",
    "Starting Ollama…": "Ollama başlatılıyor…",
    "Model is loaded and warm": "Model bellekte ve hazır",
    "Not loaded": "Bellekte değil",

    # ── tray ──────────────────────────────────────────────────────────────
    "Open Maze AI": "Maze AI'ı aç",
    "Quit": "Çıkış",
    "Still running in the tray.": "Sistem tepsisinde çalışmaya devam ediyor.",
    "⏰ Reminder": "⏰ Hatırlatma",
    # ── approval dialog ───────────────────────────────────────────────────
    "Run this command?": "Bu komut çalıştırılsın mı?",
    "Launch this application?": "Bu uygulama açılsın mı?",
    "Write this file?": "Bu dosya yazılsın mı?",
    "Apply this edit?": "Bu düzenleme uygulansın mı?",
    "Append to this file?": "Bu dosyanın sonuna eklensin mi?",
    "Delete this?": "Bu silinsin mi?",
    "Move this?": "Bu taşınsın mı?",
    "Copy this?": "Bu kopyalansın mı?",
    "Create this directory?": "Bu klasör oluşturulsun mu?",
    "Copy this to the clipboard?": "Bu, panoya kopyalansın mı?",
    "Allow this network request?": "Bu ağ isteğine izin verilsin mi?",
    "Restore this file?": "Bu dosya geri yüklensin mi?",
    "Approve this action?": "Bu işlem onaylansın mı?",
    "You can edit the command before running it.":
        "Çalıştırmadan önce komutu düzenleyebilirsiniz.",
    "Deny": "Reddet",
    "Always allow": "Her zaman izin ver",
    "Approve and never ask for this exact command again":
        "Onayla ve tam olarak bu komut için bir daha sorma",
    "Approve & Run": "Onayla ve çalıştır",
    "⚠  Asking because {reason}.": "⚠  Sorma nedeni: {reason}.",
    # approval reasons (keys come from the agent, wording lives here)
    "it downloads content and writes it to a file":
        "içeriği indirip bir dosyaya yazıyor",
    "this URL carries data out to a remote server":
        "bu adres uzak bir sunucuya veri taşıyor",
    "it touches a sensitive path ({detail})":
        "hassas bir yola dokunuyor ({detail})",
    "this command is destructive and cannot be undone":
        "bu komut yıkıcı ve geri alınamaz",
    "it runs a command on your machine": "makinenizde bir komut çalıştırıyor",
    "it changes something on your machine": "makinenizde bir şeyi değiştiriyor",
    # ── onboarding ────────────────────────────────────────────────────────
    "Welcome to Maze AI": "Maze AI'a hoş geldiniz",
    "Your private, agentic assistant for Maze Linux. It can run commands, "
    "manage files, launch apps, search the web, take screenshots and set "
    "reminders — right from this window.\n\n"
    "First, choose where the model runs:\n"
    "• **Ollama** — fully local & private (download a model)\n"
    "• **Gemini** or an **OpenAI-compatible** API — hosted, just add a key\n\n"
    "You stay in control: in **Ask** mode every action is confirmed, and "
    "destructive commands always require approval.":
        "Maze Linux için gizliliğe saygılı, iş yapabilen asistanınız. Bu "
        "pencereden komut çalıştırabilir, dosyaları yönetebilir, uygulama "
        "açabilir, web'de arama yapabilir, ekran görüntüsü alabilir ve "
        "hatırlatıcı kurabilir.\n\n"
        "Önce modelin nerede çalışacağını seçin:\n"
        "• **Ollama** — tamamen yerel ve gizli (bir model indirin)\n"
        "• **Gemini** veya **OpenAI uyumlu** bir API — sunucuda, tek gereken "
        "bir anahtar\n\n"
        "Kontrol sizde: **Sor** modunda her işlem onaylanır ve yıkıcı komutlar "
        "her zaman onay ister.",
    "Skip for now": "Şimdilik geç",
    "Choose a backend": "Bir sağlayıcı seç",
    # ── reminders dialog ──────────────────────────────────────────────────
    "Remind me to…": "Bana şunu hatırlat…",
    "in 30 minutes": "30 dakika sonra",
    "Add": "Ekle",
    "Enter what to be reminded about.": "Neyin hatırlatılacağını yazın.",
    "Couldn't read the time. Try 'in 10 minutes', '18:30', 'tomorrow 09:00'.":
        "Zamanı anlayamadım. '10 dakika sonra', '18:30' ya da 'yarın 09:00' deneyin.",
    "Remove": "Kaldır",
    "No pending reminders.": "Bekleyen hatırlatıcı yok.",
    # ── settings ──────────────────────────────────────────────────────────
    "Backends, models and agent behaviour": "Sağlayıcılar, modeller ve ajan davranışı",
    "AI Backend": "Yapay zekâ sağlayıcısı",
    "Choose where the model runs. Switch any time.":
        "Modelin nerede çalışacağını seçin. İstediğiniz zaman değiştirin.",
    "Ollama  ·  local, private models": "Ollama  ·  yerel, gizli modeller",
    "Gemini  ·  Google hosted API": "Gemini  ·  Google'ın sunucu API'si",
    "OpenAI-compatible  ·  OpenAI, OpenRouter, Groq, LM Studio…":
        "OpenAI uyumlu  ·  OpenAI, OpenRouter, Groq, LM Studio…",
    "Agent": "Ajan",
    "How much freedom the assistant has to act on your machine.":
        "Asistanın makinenizde ne kadar serbest hareket edeceği.",
    "Ask before acting  ·  recommended": "İşlemden önce sor  ·  önerilen",
    "Autonomous  ·  run tools without asking": "Otonom  ·  sormadan çalıştır",
    "Chat only  ·  no tools": "Yalnızca sohbet  ·  araç yok",
    "You approve every command, app launch and file write before it runs.":
        "Her komutu, uygulama açmayı ve dosya yazmayı önceden siz onaylarsınız.",
    "The agent runs tools on its own. Fast, but review what it does.":
        "Ajan araçları kendi başına çalıştırır. Hızlıdır, ama yaptıklarını gözden geçirin.",
    "Pure conversation — the assistant cannot touch your system.":
        "Saf sohbet — asistan sisteminize dokunamaz.",
    "RESPONSE LANGUAGE": "YANIT DİLİ",
    "INTERFACE LANGUAGE": "ARAYÜZ DİLİ",
    "Takes effect the next time Maze AI starts.":
        "Maze AI'ın bir sonraki açılışında geçerli olur.",
    "MAX STEPS": "EN FAZLA ADIM",
    "COMMAND TIMEOUT (S)": "KOMUT ZAMAN AŞIMI (SN)",
    "Behaviour & safety": "Davranış ve güvenlik",
    "Streaming, custom instructions and command guards.":
        "Akış, kalıcı talimatlar ve komut korumaları.",
    "Stream responses as they're generated": "Yanıtları üretilirken akıt",
    "Always confirm destructive commands (even in autonomous mode)":
        "Yıkıcı komutları her zaman onayla (otonom modda bile)",
    "Auto-approve safe read-only commands (skip the prompt)":
        "Güvenli, salt-okunur komutları otomatik onayla",
    "Always confirm access to keys, tokens and private history":
        "Anahtarlara, jetonlara ve özel geçmişe erişimi her zaman onayla",
    "Confirm network requests that carry data out":
        "Dışarıya veri taşıyan ağ isteklerini onayla",
    "TOOLS THE AGENT MAY USE": "AJANIN KULLANABİLECEĞİ ARAÇLAR",
    "Shell commands and launching apps": "Kabuk komutları ve uygulama açma",
    "Reading and changing files": "Dosya okuma ve değiştirme",
    "Web search and fetching pages": "Web araması ve sayfa getirme",
    "Screenshots, OCR, clipboard, notifications":
        "Ekran görüntüsü, OCR, pano, bildirimler",
    "Reminders and to-dos": "Hatırlatıcılar ve yapılacaklar",
    "No tools — the assistant can only talk.":
        "Araç yok — asistan yalnızca sohbet edebilir.",
    "{count} tools · about {tokens} tokens of context per message":
        "{count} araç · mesaj başına yaklaşık {tokens} token bağlam",
    "CUSTOM INSTRUCTIONS": "KALICI TALİMATLAR",
    "Standing instructions for every chat — e.g. “Always use the fish shell”, "
    "“Prefer concise answers”, project context…":
        "Her sohbet için geçerli talimatlar — örn. “Her zaman fish kabuğunu kullan”, "
        "“Kısa yanıtlar ver”, proje bağlamı…",
    "Window & notifications": "Pencere ve bildirimler",
    "Start Maze AI on login (in the tray)":
        "Maze AI'ı oturum açılışında başlat (tepsiye)",
    "Closing the window hides it to the system tray":
        "Pencereyi kapatmak sistem tepsisine gizler",
    "Greet me with a notification on startup":
        "Açılışta bildirimle selamla",
    "Cancel": "Vazgeç",
    "Save": "Kaydet",
    "OLLAMA HOST": "OLLAMA SUNUCUSU",
    "ACTIVE MODEL  (installed)": "ETKİN MODEL  (kurulu)",
    "List installed models": "Kurulu modelleri listele",
    "CONTEXT WINDOW  (tokens)": "BAĞLAM PENCERESİ  (token)",
    "DOWNLOAD A MODEL": "MODEL İNDİR",
    "Download": "İndir",
    "Browse more at ollama.com/library": "Daha fazlası: ollama.com/library",
    "GEMINI API KEY": "GEMINI API ANAHTARI",
    "Get a key at aistudio.google.com/apikey":
        "Anahtar alın: aistudio.google.com/apikey",
    "MODEL": "MODEL",
    "Fetch available models": "Kullanılabilir modelleri getir",
    "API BASE URL": "API TEMEL ADRESİ",
    "API KEY  (blank for most local servers)":
        "API ANAHTARI  (çoğu yerel sunucuda boş)",
    "Works with any OpenAI-compatible /v1 endpoint.":
        "OpenAI uyumlu herhangi bir /v1 uç noktasıyla çalışır.",
    "Delete this model from disk": "Bu modeli diskten sil",
    "Auto — from the model's limit and this machine's RAM":
        "Otomatik — modelin sınırına ve bu makinenin RAM'ine göre",
    "KEEP THE MODEL IN MEMORY": "MODELİ BELLEKTE TUT",
    "5 minutes": "5 dakika",
    "30 minutes": "30 dakika",
    "2 hours": "2 saat",
    "Forever (until you quit Ollama)": "Sürekli (Ollama kapanana dek)",
    "Never — unload after each reply": "Hiç — her yanıttan sonra bellekten at",
    "A cold model spends 5–15 seconds loading before its first token. "
    "Keeping it resident trades RAM for an instant reply.":
        "Soğuk bir model ilk token'dan önce 5–15 saniye yüklenmeye harcar. "
        "Bellekte tutmak, RAM karşılığında anında yanıt demektir.",
    "Load the model at startup, before the first question":
        "Modeli açılışta, ilk sorudan önce yükle",
    "Let thinking models reason first (slower, often better)":
        "Düşünen modeller önce akıl yürütsün (daha yavaş, çoğu zaman daha iyi)",
    "Use the model's own tool calling when it supports it":
        "Model destekliyorsa kendi araç çağrısını kullan",
    "Native function calling is far more reliable than asking a model to "
    "hand-write JSON, and it keeps the tool catalogue out of the context "
    "window. Falls back to the JSON protocol automatically.":
        "Yerel fonksiyon çağrısı, modele JSON yazdırmaktan çok daha güvenilirdir "
        "ve araç kataloğunu bağlam penceresinin dışında tutar. Desteklenmiyorsa "
        "otomatik olarak JSON protokolüne döner.",
    "Force valid JSON on models without tool calling":
        "Araç çağrısı olmayan modellerde geçerli JSON'u zorunlu kıl",
    "Constrains generation to the protocol schema, so a small local model "
    "cannot produce truncated or fenced JSON.":
        "Üretimi protokol şemasına bağlar; küçük bir yerel model kesik ya da "
        "kod bloğuna sarılmış JSON üretemez.",
    "context {tokens}": "bağlam {tokens}",
    "auto → {tokens}": "otomatik → {tokens}",
    "native tool calling": "yerel araç çağrısı",
    "JSON protocol (no tool calling)": "JSON protokolü (araç çağrısı yok)",
    "vision": "görme",
    "thinking": "düşünme",
    # ── hardware & placement ──────────────────────────────────────────────
    "GPU LAYERS  (0 = let Ollama decide)": "GPU KATMANLARI  (0 = Ollama karar versin)",
    "Forces how many layers are offloaded to the GPU. Leave at 0 unless "
    "Ollama's own estimate is wrong: raise it to use VRAM it left idle, "
    "lower it if loading fails with an out-of-memory error.":
        "Kaç katmanın GPU'ya aktarılacağını zorlar. Ollama'nın kendi tahmini "
        "yanlış olmadıkça 0'da bırakın: boş kalan VRAM'i kullanmak için "
        "artırın, bellek yetersizliği hatasında azaltın.",
    "Measure": "Ölç",
    "Run a short generation and report load time, prefill and "
    "generation speed, and whether the model stayed on the GPU.":
        "Kısa bir üretim çalıştırır; yükleme süresini, okuma ve yazma hızını "
        "ve modelin GPU'da kalıp kalmadığını bildirir.",
    "Unload": "Bellekten çıkar",
    "Drop the model from memory and free the VRAM":
        "Modeli bellekten atar ve VRAM'i boşaltır",
    "Fits in VRAM — will run fully on the GPU.":
        "VRAM'e sığıyor — tamamen GPU'da çalışacak.",
    "Only just fits in VRAM; other apps may push it out.":
        "VRAM'e zar zor sığıyor; başka uygulamalar taşmasına yol açabilir.",
    "Too big at this context — part of it will run on the CPU.":
        "Bu bağlamda çok büyük — bir kısmı CPU'da çalışacak.",
    "Will not fit in VRAM — this model runs on the CPU.":
        "VRAM'e sığmıyor — bu model CPU'da çalışır.",
    "Needs ~{need} GB, {usable} GB of VRAM usable":
        "~{need} GB gerekiyor, {usable} GB VRAM kullanılabilir",
    "(other apps hold {other} GB)": "(diğer uygulamalar {other} GB tutuyor)",
    "Suggested context: {tokens}": "Önerilen bağlam: {tokens}",
    "In memory now · {processor} · unloads in {minutes} min":
        "Şu an bellekte · {processor} · {minutes} dk sonra atılır",
    "Unloaded '{model}' — VRAM freed.": "'{model}' bellekten atıldı — VRAM boşaldı.",
    "Could not unload '{model}'.": "'{model}' bellekten atılamadı.",
    "Measuring '{model}'…": "'{model}' ölçülüyor…",
    "{model}: {load}s to load · {prefill} tok/s reading · "
    "{generate} tok/s writing · {processor}":
        "{model}: {load}s yükleme · okuma {prefill} tok/s · "
        "yazma {generate} tok/s · {processor}",
    "unknown": "bilinmiyor",
    "Shrink context to {tokens}": "Bağlamı {tokens} yap",
    "Part of the model is running on the CPU. A smaller context window keeps "
    "it in VRAM, which is several times faster.":
        "Modelin bir kısmı CPU'da çalışıyor. Daha küçük bir bağlam penceresi "
        "onu VRAM'de tutar ve bu kat kat hızlıdır.",
    "Context set to {tokens}. The model will reload on the next message.":
        "Bağlam {tokens} yapıldı. Model bir sonraki mesajda yeniden yüklenecek.",
    "{model} needs about {need} GB; {usable} GB of VRAM is usable, "
    "so part of it runs on the CPU.":
        "{model} yaklaşık {need} GB istiyor; kullanılabilir VRAM {usable} GB, "
        "bu yüzden bir kısmı CPU'da çalışıyor.",
    "Installed models that would fit: {models}":
        "Sığacak kurulu modeller: {models}",
    "context {used}/{limit}": "bağlam {used}/{limit}",
    "In memory now: {models}": "Şu an bellekte: {models}",
    "Click 🗑 again to delete '{model}' from disk.":
        "'{model}' modelini diskten silmek için 🗑 düğmesine tekrar basın.",
    "Deleted '{model}'.": "'{model}' silindi.",
    "Could not delete '{model}'.": "'{model}' silinemedi.",
    "Querying Ollama…": "Ollama sorgulanıyor…",
    "Querying Gemini…": "Gemini sorgulanıyor…",
    "Querying the API…": "API sorgulanıyor…",
    "Found {count} model(s).": "{count} model bulundu.",
    "No models found — is Ollama running?": "Model bulunamadı — Ollama çalışıyor mu?",
    "Enter a model name to download.": "İndirilecek model adını yazın.",
    "Contacting Ollama to download '{model}'…":
        "'{model}' indirmek için Ollama'ya bağlanılıyor…",
    "Please wait for the download to finish…": "Lütfen indirmenin bitmesini bekleyin…",
    "Connecting…": "Bağlanılıyor…",
    "Fetching manifest…": "Manifest alınıyor…",
    "Verifying…": "Doğrulanıyor…",
    "Finishing up…": "Tamamlanıyor…",
    "Cleaning up…": "Temizleniyor…",
    "Downloading…": "İndiriliyor…",
    "Done": "Bitti",
    "Done ✓": "Bitti ✓",
    "Downloading · {done}/{total} MB ({pct}%){speed}":
        "İndiriliyor · {done}/{total} MB (%{pct}){speed}",
    "✓ '{model}' downloaded and ready.": "✓ '{model}' indirildi ve hazır.",
    "How many tokens the model can hold at once. Raise this if you hit "
    "'exceeds the available context size' errors (needs more RAM/VRAM). "
    "8192 is a safe default; images and long chats need more.":
        "Modelin aynı anda tutabileceği token sayısı. 'exceeds the available "
        "context size' hatası alıyorsanız artırın (daha çok RAM/VRAM ister). "
        "8192 güvenli bir varsayılandır; görseller ve uzun sohbetler daha "
        "fazlasını gerektirir.",
    "Reading ~/.ssh, ~/.gnupg, .env files, browser profiles or shell "
    "history always asks first — in every mode. This is the step a "
    "malicious web page would need to steal your credentials.":
        "~/.ssh, ~/.gnupg, .env dosyaları, tarayıcı profilleri veya kabuk "
        "geçmişini okumak her modda önce sorar. Kötü niyetli bir web "
        "sayfasının kimlik bilgilerinizi çalmak için ihtiyaç duyduğu adım "
        "tam olarak budur.",
    "A fetch whose URL carries a payload (or that saves the download to "
    "a file) needs approval, so content the assistant read cannot talk "
    "it into sending your data somewhere.":
        "Adresinde veri taşıyan (ya da indirdiğini dosyaya kaydeden) bir "
        "istek onay ister; böylece asistanın okuduğu bir içerik onu "
        "verilerinizi bir yere göndermeye ikna edemez.",
    "Auto  ·  match my system": "Otomatik  ·  sistemimle aynı",
}

_CATALOGS: dict[str, dict[str, str]] = {"tr": TURKISH}

_current = "en"


def _system_language() -> str:
    """The desktop's language code, from the usual locale environment."""
    for var in ("LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE"):
        value = os.environ.get(var, "")
        if value:
            code = value.split(":")[0].split(".")[0].split("_")[0].strip().lower()
            if code and code not in ("c", "posix"):
                return code
    return "en"


def set_language(code: str) -> str:
    """Select the interface language. ``auto`` follows the desktop locale.

    Returns the code actually in use, so callers can report it.
    """
    global _current
    code = (code or "auto").strip().lower()
    if code == "auto":
        code = _system_language()
    _current = code if code in _CATALOGS else "en"
    return _current


def current_language() -> str:
    return _current


def tr(text: str) -> str:
    """Translate a UI string, falling back to the English source."""
    return _CATALOGS.get(_current, {}).get(text, text)
