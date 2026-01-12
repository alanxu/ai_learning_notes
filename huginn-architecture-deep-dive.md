# Huginn Architecture Deep Dive

A comprehensive analysis of Huginn's architecture, with special focus on its information source monitoring capabilities.

## Overview

**Huginn** is a Ruby on Rails-based automation platform for building autonomous agents that perform automated tasks online. It's essentially a self-hosted, hackable alternative to IFTTT or Zapier.

**Repository Location:** `/Users/alanxu/projects/huginn`

## Core Architecture

Huginn follows a classic Ruby on Rails MVC architecture with some key abstractions:

```
┌──────────────────────────────────────────────────────────────────┐
│                         User Interface                            │
│                   (Rails Views + JavaScript)                      │
└──────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────────────────────────────────────────┐
│                          Controllers                              │
│        AgentsController, EventsController, ScenariosController    │
└──────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────────────────────────────────────────┐
│                         Core Models                               │
│                    Agent ─────── Event                            │
│                      │             │                              │
│              ┌───────┴───────┐     │                              │
│           Link          ControlLink                               │
│              │               │                                    │
│           Scenario    ScenarioMembership                          │
└──────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────────────────────────────────────────┐
│                     Background Processing                         │
│         HuginnScheduler + Delayed Job + Rufus Scheduler          │
└──────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
huginn/
├── app/
│   ├── models/
│   │   ├── agent.rb           # Base Agent class
│   │   ├── agents/            # 60+ specialized agent types
│   │   ├── event.rb           # Event model
│   │   ├── link.rb            # Agent-to-agent links
│   │   ├── control_link.rb    # Agent control relationships
│   │   └── scenario.rb        # Agent groupings
│   ├── concerns/              # Shared behavior modules
│   ├── controllers/           # HTTP handlers
│   ├── jobs/                  # Background jobs
│   └── views/                 # UI templates
├── lib/
│   ├── huginn_scheduler.rb    # Scheduling infrastructure
│   └── agent_runner.rb        # Agent execution manager
├── config/                    # Rails configuration
└── db/migrate/                # Database migrations (76+)
```

---

## The Agent System

### Base Agent Model (`app/models/agent.rb`)

The `Agent` class is the heart of Huginn. Every agent type inherits from this base class.

**Key Features:**

```ruby
class Agent < ActiveRecord::Base
  include LiquidInterpolatable   # Template variable support
  include WebRequestConcern      # HTTP request capabilities
  include DryRunnable            # Test execution support
  include WorkingHelpers         # Health/status tracking

  json_serialize :options, :memory  # Persistent configuration & state

  # Core API methods that subclasses implement:
  def check           # Called on schedule to poll data sources
  def receive(events) # Called when linked agents emit events
  def working?        # Returns health status
  def default_options # Default configuration
end
```

**Scheduling Options:**
```ruby
SCHEDULES = %w[
  every_1m every_2m every_5m every_10m every_30m
  every_1h every_2h every_5h every_12h
  every_1d every_2d every_7d
  midnight 1am 2am ... 11pm
  never
]
```

### Agent Relationships

Huginn uses a **directed graph** model for agent communication:

```
┌─────────────┐     Link      ┌─────────────┐     Link      ┌─────────────┐
│   Source    │──────────────►│  Processor  │──────────────►│   Output    │
│    Agent    │   (events)    │    Agent    │   (events)    │    Agent    │
└─────────────┘               └─────────────┘               └─────────────┘
                                    │
                             ControlLink
                                    ▼
                            ┌─────────────┐
                            │   Target    │
                            │    Agent    │
                            └─────────────┘
```

- **Link**: Defines event flow (source emits → receiver receives)
- **ControlLink**: Allows agents to enable/disable other agents

### Agent Lifecycle

1. **Scheduled Check**: `HuginnScheduler` triggers `Agent.run_schedule(schedule_name)`
2. **Bulk Check**: `AgentCheckJob` queued for each agent
3. **Execution**: Agent's `check()` method called
4. **Event Creation**: Agent calls `create_event(payload: {...})`
5. **Propagation**: `AgentPropagateJob` routes events to linked receivers
6. **Reception**: Receiver's `receive(events)` method called

---

## Information Source Monitoring Design

Huginn's power lies in its flexible approach to monitoring diverse information sources. Here's how it's designed:

### Pattern 1: Polling (Scheduled Check)

Most agents use a **polling pattern** - periodically checking sources on a schedule.

**Example: WebsiteAgent** (`app/models/agents/website_agent.rb`)

```ruby
module Agents
  class WebsiteAgent < Agent
    include WebRequestConcern    # Gives HTTP capabilities

    can_dry_run!
    default_schedule "every_12h"

    def check
      check_urls(interpolated['url'])  # Fetch & parse on schedule
    end

    def check_url(url, existing_payload = {})
      response = faraday.get(uri)      # HTTP GET via Faraday
      handle_data(response.body, response.env[:url], existing_payload)
    end

    def handle_data(body, url, existing_payload)
      doc = parse(body)                # Parse HTML/XML/JSON/Text
      output = extract_xml(doc)        # Extract data via CSS/XPath

      output.each do |extracted|
        if store_payload!(old_events, result)  # Deduplication
          create_event payload: existing_payload.merge(result)
        end
      end
    end
  end
end
```

**Key Capabilities:**
- Supports **HTML, XML, JSON, and text** parsing
- Uses **CSS selectors** or **XPath expressions** for data extraction
- Built-in **change detection** (`on_change` mode)
- Handles **multiple URLs** in a single agent
- **Event-driven scraping**: Can receive events with URLs to scrape

### Pattern 2: Feed Parsing (RSS/Atom)

**Example: RssAgent** (`app/models/agents/rss_agent.rb`)

```ruby
module Agents
  class RssAgent < Agent
    include WebRequestConcern

    cannot_receive_events!      # Source-only agent
    can_dry_run!
    default_schedule "every_1d"

    gem_dependency_check { defined?(Feedjira) }  # External dependency

    def check
      check_urls(Array(interpolated['url']))
    end

    def check_urls(urls)
      urls.each do |url|
        response = faraday.get(url)
        feed = Feedjira.parse(preprocessed_body(response))
        new_events.concat feed_to_events(feed)
      end

      # Deduplication via memory
      events = sort_events(new_events).select { |event, index|
        check_and_track(event.payload[:id])  # Track seen IDs
      }
      create_events(events)
    end

    def check_and_track(entry_id)
      memory['seen_ids'] ||= []
      return false if memory['seen_ids'].include?(entry_id)
      memory['seen_ids'].unshift entry_id
      memory['seen_ids'].pop if memory['seen_ids'].length > remembered_id_count
      true
    end
  end
end
```

**Key Capabilities:**
- Uses **Feedjira** library for robust feed parsing
- Handles **RSS and Atom** formats
- **iTunes podcast** support with specialized fields
- **Deduplication** via persistent memory (stores last 500 IDs)
- **Sorting** by publication date

### Pattern 3: Push-Based (Webhooks)

**Example: WebhookAgent** (`app/models/agents/webhook_agent.rb`)

```ruby
module Agents
  class WebhookAgent < Agent
    cannot_be_scheduled!        # No polling - receives pushes
    cannot_receive_events!

    def receive_web_request(request)
      secret = request.path_parameters[:secret]
      return ["Not Authorized", 401] unless secret == interpolated['secret']

      params = request.query_parameters.dup
      params.update(request.request_parameters)

      [payload_for(params)].flatten.each do |payload|
        create_event(payload: payload.merge(event_headers_payload(headers)))
      end

      [interpolated['response'] || 'Event Created', 201]
    end
  end
end
```

**Endpoint Format:**
```
POST https://domain/users/{user_id}/web_requests/{agent_id}/{secret}
```

**Key Capabilities:**
- **Secret-based authentication**
- **Configurable HTTP verbs** (GET, POST, etc.)
- **JSONPath extraction** from request body
- **reCAPTCHA integration** for form submissions
- **Custom response** codes and headers

### Pattern 4: Protocol-Specific Clients (IMAP)

**Example: ImapFolderAgent** (`app/models/agents/imap_folder_agent.rb`)

```ruby
module Agents
  class ImapFolderAgent < Agent
    include GoogleOauth2Concern  # OAuth support for Gmail

    cannot_receive_events!
    default_schedule "every_30m"

    def check
      each_unread_mail { |mail, notified|
        # Match conditions against email
        interpolated['conditions'].all? { |key, value|
          case key
          when 'subject'
            re = Regexp.new(value)
            re.match(mail.scrubbed(:subject))
          when 'body'
            # Match against body parts
          when 'from', 'to', 'cc'
            # Glob pattern matching
          end
        } or next

        create_event(payload: {
          'message_id' => message_id,
          'folder' => mail.folder,
          'subject' => mail.scrubbed(:subject),
          'body' => body,
          # ... more fields
        })
      }
    end

    def each_unread_mail
      Client.open(host, port: port, ssl: ssl) { |imap|
        imap.login(username, password)

        interpolated['folders'].each { |folder|
          imap.select(folder)
          # Track UID validity to detect mailbox changes
          # Fetch only new messages since last check
          uids = imap.uid_fetch((lastseenuid + 1)..-1, 'FLAGS')
          # ...
        }
      }
    end
  end
end
```

**Key Capabilities:**
- **IMAP protocol** implementation via Net::IMAP
- **Gmail OAuth2** support
- **Multiple folder** monitoring
- **Regex-based** subject/body filtering
- **Glob patterns** for address matching
- **UID tracking** to avoid reprocessing
- **Mark as read / delete** actions

### Pattern 5: File System Monitoring

**Example: LocalFileAgent** (`app/models/agents/local_file_agent.rb`)

```ruby
module Agents
  class LocalFileAgent < Agent
    include LongRunnable    # Enables background worker

    def start_worker?
      interpolated['mode'] == 'read' && boolify(interpolated['watch'])
    end

    class Worker < LongRunnable::Worker
      def setup
        require 'listen'    # File watching library
        @listener = Listen.to(path, **options, &method(:callback))
      end

      def run
        @listener.start
        sleep              # Keep running indefinitely
      end

      def callback(*changes)
        changes.zip([:modified, :added, :removed]).each do |files, event_type|
          files.each do |file|
            agent.create_event payload: {
              file_pointer: { file: file, agent_id: agent.id },
              event_type: event_type
            }
          end
        end
      end
    end
  end
end
```

**Key Capabilities:**
- **Real-time file watching** via Listen gem
- **Directory or single file** monitoring
- **Event types**: modified, added, removed
- **Background worker** pattern for continuous monitoring

---

## Event System

### Event Model (`app/models/event.rb`)

```ruby
class Event < ActiveRecord::Base
  json_serialize :payload    # Arbitrary JSON data

  belongs_to :agent
  has_many :agent_logs_as_inbound_event
  has_many :agent_logs_as_outbound_event

  # Location tracking
  def location
    Location.new(lat: lat, lng: lng, ...)
  end

  # Event expiration
  scope :expired, -> {
    where("expires_at IS NOT NULL AND expires_at < ?", Time.now)
  }
end
```

**Event Lifecycle:**
1. Agent creates event via `create_event(payload: {...})`
2. Event saved with `expires_at` based on agent's `keep_events_for` setting
3. `possibly_propagate` triggers immediate propagation if receivers want it
4. `AgentPropagateJob` distributes events to linked agents
5. `AgentCleanupExpiredJob` removes expired events

### Event Propagation

Two modes:
1. **Deferred** (default): Events batched and propagated every minute
2. **Immediate**: When receiver has `propagate_immediately: true`

```ruby
# From Agent.receive!
def self.receive!(options = {})
  agents_to_events = {}

  # Find all agents with new events from their sources
  Agent.select(...)
    .joins("JOIN links ON ...")
    .joins("JOIN events ON ...")
    .where("events.id > agents.last_checked_event_id")

  # Queue receive jobs
  agents_to_events.each do |agent_id, event_ids|
    Agent.async_receive(agent_id, event_ids)
  end
end
```

---

## Concerns (Shared Behavior)

### WebRequestConcern (`app/concerns/web_request_concern.rb`)

Provides HTTP capabilities to agents:

```ruby
module WebRequestConcern
  def faraday
    @faraday ||= Faraday.new(faraday_options) { |builder|
      builder.response :character_encoding  # Handle encoding
      builder.response :follow_redirects    # Follow redirects
      builder.request :multipart            # File uploads
      builder.request :url_encoded          # Form data
      builder.request :gzip                 # Compression

      if userinfo = basic_auth_credentials
        builder.request :authorization, :basic, *userinfo
      end
    }
  end
end
```

**Configurable Options:**
- `user_agent`: Custom User-Agent string
- `headers`: Custom HTTP headers
- `basic_auth`: HTTP basic authentication
- `proxy`: Proxy server
- `disable_ssl_verification`: Skip SSL checks
- `force_encoding`: Override character encoding

### LiquidInterpolatable (`app/concerns/liquid_interpolatable.rb`)

Enables dynamic variable interpolation in agent options:

```ruby
module LiquidInterpolatable
  def interpolated(self_object = nil)
    interpolate_options(options)
  end

  def interpolate_string(string)
    Liquid::Template.parse(string).render!(interpolation_context)
  end
end
```

**Available Variables:**
- `{{ event.field }}` - Event payload fields
- `{{ _agent_.name }}` - Agent properties
- `{% credential token_name %}` - Stored credentials

**Built-in Filters:**
- `uri_escape`, `to_uri`, `uri_expand`
- `regex_extract`, `regex_replace`
- `json`, `fromjson`
- `md5`, `sha1`, `sha256`, `hmac_sha1`, `hmac_sha256`
- `as_object` - Return parsed data structure

---

## Scheduling Infrastructure

### HuginnScheduler (`lib/huginn_scheduler.rb`)

Uses **Rufus Scheduler** for cron-like scheduling:

```ruby
class HuginnScheduler < LongRunnable::Worker
  SCHEDULE_TO_CRON = {
    '1m'  => '*/1 * * * *',
    '5m'  => '*/5 * * * *',
    '1h'  => '0 * * * *',
    '1d'  => '0 0 * * *',
    # ...
  }

  def setup
    # Propagate events every minute
    every '1m' do
      propagate!
    end

    # Clean up expired events
    every '6h' do
      cleanup_expired_events!
    end

    # Schedule each time interval
    SCHEDULE_TO_CRON.keys.each do |schedule|
      cron SCHEDULE_TO_CRON[schedule] do
        run_schedule "every_#{schedule}"
      end
    end

    # Schedule specific times (midnight, 1am, etc.)
    24.times do |hour|
      cron "0 #{hour} * * *" do
        run_schedule hour_to_schedule_name(hour)
      end
    end
  end
end
```

### Background Jobs

All agent work is processed via **Delayed Job**:

| Job | Purpose |
|-----|---------|
| `AgentCheckJob` | Execute agent's `check()` method |
| `AgentReceiveJob` | Route events to agent's `receive()` |
| `AgentPropagateJob` | Find and distribute new events |
| `AgentRunScheduleJob` | Trigger scheduled agent checks |
| `AgentCleanupExpiredJob` | Remove expired events |

---

## Agent Categories

### Data Source Agents (Input)

| Agent | Source Type | Monitoring Method |
|-------|-------------|-------------------|
| `WebsiteAgent` | Web pages, APIs | HTTP polling + parsing |
| `RssAgent` | RSS/Atom feeds | Feed parsing |
| `WebhookAgent` | External services | Push via HTTP |
| `ImapFolderAgent` | Email | IMAP polling |
| `LocalFileAgent` | File system | File watching |
| `TwitterSearchAgent` | Twitter | API polling |
| `WeatherAgent` | Weather APIs | API polling |
| `HttpStatusAgent` | Web endpoints | HTTP HEAD requests |
| `S3Agent` | AWS S3 | S3 API polling |

### Data Processing Agents

| Agent | Function |
|-------|----------|
| `EventFormattingAgent` | Transform event payloads |
| `JsonParseAgent` | Parse JSON strings |
| `JqAgent` | jq-style queries |
| `JavaScriptAgent` | Custom JS logic |
| `TriggerAgent` | Conditional filtering |
| `DeDuplicationAgent` | Remove duplicates |
| `ChangeDetectorAgent` | Detect value changes |
| `DigestAgent` | Aggregate events |
| `DelayAgent` | Time-based delays |

### Output/Action Agents

| Agent | Action |
|-------|--------|
| `EmailAgent` | Send emails |
| `SlackAgent` | Post to Slack |
| `TwilioAgent` | Send SMS |
| `PostAgent` | HTTP POST requests |
| `DataOutputAgent` | Generate RSS/JSON feeds |
| `ShellCommandAgent` | Execute shell commands |
| `PushbulletAgent` | Push notifications |

---

## Key Design Patterns

### 1. Template Method Pattern
Base `Agent` class defines the lifecycle; subclasses implement specifics:
```ruby
def check     # Override to poll data sources
def receive   # Override to handle incoming events
def working?  # Override to report health status
```

### 2. Concern Composition
Shared functionality mixed into agents:
```ruby
class WebsiteAgent < Agent
  include WebRequestConcern      # HTTP
  include LiquidInterpolatable   # Templates
  include DryRunnable            # Testing
end
```

### 3. Memory Persistence
Agents maintain state across runs via serialized `memory` hash:
```ruby
memory['seen_ids'] ||= []
memory['last_value'] = current_value
save!
```

### 4. Event-Driven Architecture
Loose coupling via events:
```ruby
# Producer
create_event(payload: { data: "..." })

# Consumer (linked agent)
def receive(events)
  events.each { |event| process(event.payload) }
end
```

### 5. Deduplication Strategies
- **WebsiteAgent**: Compare payload JSON, update expiration for matches
- **RssAgent**: Track seen entry IDs in memory
- **ImapFolderAgent**: Track UIDs and Message-IDs

---

## Configuration & Deployment

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `DOMAIN` | Public domain for webhooks |
| `DATABASE_URL` | Database connection |
| `TIMEZONE` | Scheduler timezone |
| `ENABLE_INSECURE_AGENTS` | Enable file system agents |
| `DEFAULT_HTTP_USER_AGENT` | Default User-Agent |
| `FARADAY_HTTP_BACKEND` | HTTP library (typhoeus/net_http) |

### Deployment Options

- **Docker**: Official image with docker-compose
- **Heroku**: One-click deploy button
- **OpenShift**: MySQL and PostgreSQL templates
- **Manual**: Capistrano deployment scripts

---

## Summary

Huginn's architecture excels at:

1. **Flexibility**: 60+ agent types covering diverse data sources
2. **Composability**: Agent linking creates complex workflows
3. **Reliability**: Persistent memory, event expiration, deduplication
4. **Extensibility**: Clear patterns for adding new agent types
5. **Scalability**: Background job processing via Delayed Job

The information source monitoring design is particularly elegant:
- **Polling** agents use scheduled `check()` methods
- **Push** agents implement `receive_web_request()`
- **Long-running** agents use the `LongRunnable` worker pattern
- All share common concerns like `WebRequestConcern` for HTTP handling

This architecture enables users to build sophisticated automation pipelines that monitor and react to changes across the web, email, files, and custom data sources.
