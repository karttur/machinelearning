---
layout: page
title: Machine learning setup
excerpt: "An archive of machine learning manuals sorted by date."
image: tropwetlandmap
search_omit: true
---

<ul class="post-list">
{% for post in site.categories.machinelearning %}
  <li><ide-setup><a href="{{ site.url }}{{ post.url }}">{{ post.title }} <span class="entry-date"><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%B %d, %Y" }}</time></span>{% if post.excerpt %} <span class="excerpt">{{ post.excerpt | remove: '\[ ... \]' | remove: '\( ... \)' | markdownify | strip_html | strip_newlines | escape_once }}</span>{% endif %}</a></ide-setup></li>
{% endfor %}
</ul>
