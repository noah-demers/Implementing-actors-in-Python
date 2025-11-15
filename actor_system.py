"""
Actor Model Implementation in Python
Author: Noah deMers
Date: November 2025

This implements a simple actor-based concurrent system where actors communicate
via asynchronous message passing. Each actor runs in its own thread with a 
thread-safe message queue.

Key Features:
- Thread-safe message queues for each actor
- Independent threads processing messages asynchronously
- Three different actor types with distinct behaviors
- Non-blocking message sending
- Comprehensive test scenarios demonstrating asynchronous behavior
"""

import threading
import queue
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod


# ============================================================================
# MESSAGE CLASS
# ============================================================================

@dataclass
class Message:
    """
    Represents a message sent between actors.
    
    Each message contains:
    - msg_type: String identifying what kind of message this is
    - value: The data payload (int, str, or Actor reference)
    
    This encapsulation allows actors to handle different message types
    and route behavior based on the message type string.
    """
    msg_type: str
    value: Any
    
    def __repr__(self):
        """Pretty print messages for debugging"""
        return f"Message('{self.msg_type}', {self.value})"


# ============================================================================
# BASE ACTOR CLASS
# ============================================================================

class Actor(ABC):
    """
    Base class for all actors in the system.
    
    An actor is a concurrent entity that:
    1. Has a thread-safe message queue
    2. Runs in its own thread
    3. Processes messages asynchronously
    4. Can send messages to other actors
    
    The actor model provides:
    - Isolation: Each actor has its own state
    - Asynchrony: Sending messages is non-blocking
    - Concurrency: Multiple actors run simultaneously
    """
    
    def __init__(self, name: str):
        """
        Initialize an actor with:
        - name: Identifier for this actor (useful for debugging/logging)
        - message_queue: Thread-safe FIFO queue for incoming messages
        - thread: Dedicated thread that processes messages
        - running: Flag to control the actor's lifecycle
        - internal_state: Dictionary for storing actor-specific data
        """
        self.name = name
        # queue.Queue is thread-safe by default in Python
        # This ensures enqueue/dequeue operations are atomic
        self.message_queue = queue.Queue()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.running = False
        # Each actor can store values for later use
        self.internal_state: Dict[str, Any] = {}
        
    def start(self):
        """
        Start the actor's thread.
        
        This begins the actor's message processing loop.
        Once started, the actor will continuously check for
        and process messages until stopped.
        """
        self.running = True
        self.thread.start()
        print(f"[{self.name}] Actor started")
        
    def stop(self):
        """
        Stop the actor's thread gracefully.
        
        Sets the running flag to False, which will cause the
        run() loop to exit. The thread will finish after processing
        its current message.
        """
        self.running = False
        # Send a poison pill message to unblock the queue if waiting
        self.send(Message("STOP", None))
        
    def send(self, message: Message):
        """
        Send a message to this actor (non-blocking).
        
        This is the fundamental communication primitive in the actor model.
        The sender puts a message in the receiver's queue and immediately
        continues without waiting for processing.
        
        Key property: ASYNCHRONOUS - sender doesn't wait for receiver
        """
        self.message_queue.put(message)
        print(f"[{self.name}] Received message: {message}")
        
    def run(self):
        """
        Main message processing loop (runs in dedicated thread).
        
        This loop:
        1. Waits for messages (blocking if queue is empty)
        2. Retrieves next message from queue
        3. Dispatches to handle_message() for processing
        4. Repeats until actor is stopped
        
        The blocking get() means the actor sleeps when idle,
        conserving CPU resources.
        """
        print(f"[{self.name}] Starting message loop")
        while self.running:
            try:
                # Block until a message arrives (with timeout for responsiveness)
                message = self.message_queue.get(timeout=0.1)
                
                # Stop message terminates the actor
                if message.msg_type == "STOP":
                    print(f"[{self.name}] Stopping")
                    break
                    
                # Delegate actual processing to subclass
                self.handle_message(message)
                
            except queue.Empty:
                # Timeout expired with no message - just continue
                continue
            except Exception as e:
                print(f"[{self.name}] Error processing message: {e}")
        
        print(f"[{self.name}] Message loop ended")
    
    @abstractmethod
    def handle_message(self, message: Message):
        """
        Process a message (implemented by each actor type).
        
        This is where actor-specific behavior is defined.
        Each actor type handles messages differently based on:
        - Message type
        - Message value
        - Current internal state
        
        This method must be implemented by all actor subclasses.
        """
        pass


# ============================================================================
# ACTOR TYPE 1: COUNTER ACTOR
# ============================================================================

class CounterActor(Actor):
    """
    An actor that stores and manipulates numeric values.
    
    Capabilities:
    1. STORE: Save a number for later use
    2. INCREMENT: Add to stored value and send to another actor
    3. GET: Retrieve current stored value
    
    This demonstrates:
    - State management (storing values)
    - Arithmetic operations
    - Sending computed results to other actors
    """
    
    def __init__(self, name: str):
        """Initialize with default counter value of 0"""
        super().__init__(name)
        self.internal_state['counter'] = 0
        
    def handle_message(self, message: Message):
        """
        Handle incoming messages based on type.
        
        Message types supported:
        - STORE: Store the value as current counter
        - INCREMENT: Add value to counter and send to target actor
        - GET: Print current counter value
        """
        msg_type = message.msg_type
        value = message.value
        
        if msg_type == "STORE":
            # Store a numeric value
            self.internal_state['counter'] = value
            print(f"[{self.name}] Stored value: {value}")
            
        elif msg_type == "INCREMENT":
            # Increment counter and send to another actor
            # Value should be tuple: (amount, target_actor)
            if isinstance(value, tuple) and len(value) == 2:
                amount, target_actor = value
                self.internal_state['counter'] += amount
                new_value = self.internal_state['counter']
                print(f"[{self.name}] Incremented by {amount}, now {new_value}")
                
                # Send incremented value to target actor
                if isinstance(target_actor, Actor):
                    target_actor.send(Message("RESULT", new_value))
                    print(f"[{self.name}] Sent {new_value} to {target_actor.name}")
            else:
                print(f"[{self.name}] Invalid INCREMENT message format")
                
        elif msg_type == "GET":
            # Display current value
            current = self.internal_state['counter']
            print(f"[{self.name}] Current counter value: {current}")
            
        else:
            print(f"[{self.name}] Unknown message type: {msg_type}")


# ============================================================================
# ACTOR TYPE 2: STRING SWITCHER ACTOR
# ============================================================================

class StringSwitcherActor(Actor):
    """
    An actor that manipulates strings and message types.
    
    Capabilities:
    1. STORE: Save a string for later use
    2. SWITCH: Swap message type and string value, send to target
    3. REVERSE: Reverse stored string and send to target
    
    This demonstrates:
    - String manipulation
    - Message transformation
    - Dynamic message creation
    """
    
    def __init__(self, name: str):
        """Initialize with empty stored string"""
        super().__init__(name)
        self.internal_state['stored_string'] = ""
        
    def handle_message(self, message: Message):
        """
        Handle incoming messages based on type.
        
        Message types supported:
        - STORE: Store string for later use
        - SWITCH: Swap msg_type and value, send to target actor
        - REVERSE: Reverse stored string and send to target
        """
        msg_type = message.msg_type
        value = message.value
        
        if msg_type == "STORE":
            # Store a string value
            self.internal_state['stored_string'] = str(value)
            print(f"[{self.name}] Stored string: '{value}'")
            
        elif msg_type == "SWITCH":
            # Switch message name and value, send to target
            # Value should be tuple: (string_value, target_actor)
            if isinstance(value, tuple) and len(value) == 2:
                string_val, target_actor = value
                # Create new message with switched fields
                new_msg = Message(str(string_val), msg_type)
                print(f"[{self.name}] Switching: '{msg_type}' <-> '{string_val}'")
                
                if isinstance(target_actor, Actor):
                    target_actor.send(new_msg)
                    print(f"[{self.name}] Sent switched message to {target_actor.name}")
            else:
                print(f"[{self.name}] Invalid SWITCH message format")
                
        elif msg_type == "REVERSE":
            # Reverse stored string and send to target actor
            if isinstance(value, Actor):
                reversed_str = self.internal_state['stored_string'][::-1]
                print(f"[{self.name}] Reversing: '{self.internal_state['stored_string']}' -> '{reversed_str}'")
                value.send(Message("RESULT", reversed_str))
                print(f"[{self.name}] Sent reversed string to {value.name}")
            else:
                print(f"[{self.name}] REVERSE requires target actor")
                
        else:
            print(f"[{self.name}] Unknown message type: {msg_type}")


# ============================================================================
# ACTOR TYPE 3: SPAWNER ACTOR
# ============================================================================

class SpawnerActor(Actor):
    """
    An actor that creates new actors dynamically.
    
    Capabilities:
    1. SPAWN: Create N new CounterActors and send them messages
    2. BROADCAST: Send same message to all spawned actors
    3. COUNT: Report how many actors have been spawned
    
    This demonstrates:
    - Dynamic actor creation
    - Managing collections of actors
    - Broadcasting messages to multiple actors
    """
    
    def __init__(self, name: str):
        """Initialize with empty list of spawned actors"""
        super().__init__(name)
        self.internal_state['spawned_actors'] = []
        
    def handle_message(self, message: Message):
        """
        Handle incoming messages based on type.
        
        Message types supported:
        - SPAWN: Create N new actors and send them initial messages
        - BROADCAST: Send message to all spawned actors
        - COUNT: Report number of spawned actors
        """
        msg_type = message.msg_type
        value = message.value
        
        if msg_type == "SPAWN":
            # Create multiple new actors
            # Value is the number of actors to create
            if isinstance(value, int) and value > 0:
                print(f"[{self.name}] Spawning {value} new actors...")
                
                for i in range(value):
                    # Create new CounterActor
                    new_actor = CounterActor(f"{self.name}_Child_{i}")
                    new_actor.start()
                    
                    # Store reference
                    self.internal_state['spawned_actors'].append(new_actor)
                    
                    # Send initial message to each new actor
                    new_actor.send(Message("STORE", i * 10))
                    
                print(f"[{self.name}] Spawned {value} actors successfully")
            else:
                print(f"[{self.name}] Invalid SPAWN value: {value}")
                
        elif msg_type == "BROADCAST":
            # Send message to all spawned actors
            # Value is tuple: (msg_type, msg_value)
            if isinstance(value, tuple) and len(value) == 2:
                broadcast_msg_type, broadcast_value = value
                spawned = self.internal_state['spawned_actors']
                print(f"[{self.name}] Broadcasting to {len(spawned)} actors")
                
                for actor in spawned:
                    actor.send(Message(broadcast_msg_type, broadcast_value))
                    
                print(f"[{self.name}] Broadcast complete")
            else:
                print(f"[{self.name}] Invalid BROADCAST format")
                
        elif msg_type == "COUNT":
            # Report number of spawned actors
            count = len(self.internal_state['spawned_actors'])
            print(f"[{self.name}] Total spawned actors: {count}")
            
        else:
            print(f"[{self.name}] Unknown message type: {msg_type}")


# ============================================================================
# ACTOR TYPE 4: PRINTER ACTOR (Helper for testing)
# ============================================================================

class PrinterActor(Actor):
    """
    Simple actor that prints received messages.
    
    Used for testing to display results from other actors.
    Demonstrates basic message reception without complex processing.
    """
    
    def __init__(self, name: str):
        """Initialize printer actor"""
        super().__init__(name)
        
    def handle_message(self, message: Message):
        """Print any received message"""
        print(f"[{self.name}] *** RECEIVED: {message} ***")


# ============================================================================
# TEST SCENARIOS
# ============================================================================

def test_basic_counter():
    """
    Test 1: Basic counter operations
    
    Tests:
    - Storing values
    - Incrementing values
    - Sending results to other actors
    """
    print("\n" + "="*70)
    print("TEST 1: Basic Counter Operations")
    print("="*70)
    
    # Create actors
    counter = CounterActor("Counter")
    printer = PrinterActor("Printer")
    
    # Start actors
    counter.start()
    printer.start()
    
    # Test sequence
    print("\n--- Storing initial value ---")
    counter.send(Message("STORE", 10))
    time.sleep(0.2)
    
    print("\n--- Incrementing and sending to printer ---")
    counter.send(Message("INCREMENT", (5, printer)))
    time.sleep(0.2)
    
    print("\n--- Checking current value ---")
    counter.send(Message("GET", None))
    time.sleep(0.2)
    
    # Cleanup
    counter.stop()
    printer.stop()
    time.sleep(0.3)


def test_string_switching():
    """
    Test 2: String manipulation and switching
    
    Tests:
    - String storage
    - Message field swapping
    - String reversal
    """
    print("\n" + "="*70)
    print("TEST 2: String Switching Operations")
    print("="*70)
    
    # Create actors
    switcher = StringSwitcherActor("Switcher")
    printer = PrinterActor("Printer")
    
    # Start actors
    switcher.start()
    printer.start()
    
    # Test sequence
    print("\n--- Storing string ---")
    switcher.send(Message("STORE", "Hello World"))
    time.sleep(0.2)
    
    print("\n--- Switching message fields ---")
    switcher.send(Message("SWITCH", ("NewType", printer)))
    time.sleep(0.2)
    
    print("\n--- Reversing string ---")
    switcher.send(Message("REVERSE", printer))
    time.sleep(0.2)
    
    # Cleanup
    switcher.stop()
    printer.stop()
    time.sleep(0.3)


def test_spawner():
    """
    Test 3: Dynamic actor creation
    
    Tests:
    - Creating multiple actors at runtime
    - Broadcasting messages to all spawned actors
    - Counting spawned actors
    """
    print("\n" + "="*70)
    print("TEST 3: Dynamic Actor Spawning")
    print("="*70)
    
    # Create spawner
    spawner = SpawnerActor("Spawner")
    spawner.start()
    
    # Test sequence
    print("\n--- Spawning 3 actors ---")
    spawner.send(Message("SPAWN", 3))
    time.sleep(0.5)  # Give time for actors to start
    
    print("\n--- Broadcasting GET message ---")
    spawner.send(Message("BROADCAST", ("GET", None)))
    time.sleep(0.3)
    
    print("\n--- Counting spawned actors ---")
    spawner.send(Message("COUNT", None))
    time.sleep(0.2)
    
    # Cleanup
    spawner.stop()
    time.sleep(0.3)


def test_asynchronous_behavior():
    """
    Test 4: Demonstrate asynchronous message handling
    
    This test shows that:
    - Messages are processed concurrently
    - Sending is non-blocking
    - Actors work independently
    
    We send multiple messages quickly and observe interleaved processing.
    """
    print("\n" + "="*70)
    print("TEST 4: Asynchronous Behavior Demonstration")
    print("="*70)
    
    # Create multiple actors
    counter1 = CounterActor("Counter1")
    counter2 = CounterActor("Counter2")
    switcher = StringSwitcherActor("Switcher")
    printer = PrinterActor("Printer")
    
    # Start all actors
    for actor in [counter1, counter2, switcher, printer]:
        actor.start()
    
    # Send many messages rapidly without waiting
    print("\n--- Sending multiple messages without blocking ---")
    start_time = time.time()
    
    counter1.send(Message("STORE", 100))
    counter2.send(Message("STORE", 200))
    switcher.send(Message("STORE", "Async"))
    counter1.send(Message("INCREMENT", (10, printer)))
    counter2.send(Message("INCREMENT", (20, printer)))
    switcher.send(Message("REVERSE", printer))
    counter1.send(Message("GET", None))
    counter2.send(Message("GET", None))
    
    send_time = time.time() - start_time
    print(f"\n--- All messages sent in {send_time:.4f} seconds (non-blocking!) ---")
    print("--- Now waiting for processing to complete ---")
    
    # Wait for all processing to complete
    time.sleep(1.0)
    
    # Cleanup
    for actor in [counter1, counter2, switcher, printer]:
        actor.stop()
    time.sleep(0.3)


def test_chain_communication():
    """
    Test 5: Actor chain communication
    
    Tests:
    - Messages passed through multiple actors
    - Actor references as message values
    - Complex communication patterns
    """
    print("\n" + "="*70)
    print("TEST 5: Chain Communication")
    print("="*70)
    
    # Create chain of actors
    counter1 = CounterActor("Counter1")
    counter2 = CounterActor("Counter2")
    counter3 = CounterActor("Counter3")
    printer = PrinterActor("Printer")
    
    # Start all
    for actor in [counter1, counter2, counter3, printer]:
        actor.start()
    
    # Set up chain: counter1 -> counter2 -> counter3 -> printer
    print("\n--- Setting up chain communication ---")
    counter1.send(Message("STORE", 5))
    time.sleep(0.1)
    
    print("\n--- Counter1 increments and sends to Counter2 ---")
    counter1.send(Message("INCREMENT", (10, counter2)))
    time.sleep(0.2)
    
    print("\n--- Counter2 increments and sends to Counter3 ---")
    counter2.send(Message("INCREMENT", (20, counter3)))
    time.sleep(0.2)
    
    print("\n--- Counter3 increments and sends to Printer ---")
    counter3.send(Message("INCREMENT", (30, printer)))
    time.sleep(0.3)
    
    # Cleanup
    for actor in [counter1, counter2, counter3, printer]:
        actor.stop()
    time.sleep(0.3)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_tests():
    """Run all test scenarios"""
    print("\n" + "#"*70)
    print("# ACTOR MODEL IMPLEMENTATION - TEST SUITE")
    print("#"*70)
    
    test_basic_counter()
    test_string_switching()
    test_spawner()
    test_asynchronous_behavior()
    test_chain_communication()
    
    print("\n" + "#"*70)
    print("# ALL TESTS COMPLETED")
    print("#"*70)


if __name__ == "__main__":
    run_all_tests()
